#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifdef PICO_BUILD
#include "pico/stdlib.h"
#include "hardware/watchdog.h"  /* watchdog_enable() used by 'reboot' command */
#endif

#include "weights_mnist.h"   /* trained Q8.8 weight arrays for all layers */
#include "test_image_q88.h"

volatile int32_t FI_SUM = 0;
volatile int FI_PREDICTION = -1;

/* =========================================================================
 * Configuration — must match the values used when weights were exported
 * ========================================================================= */
#define INPUT_H       32     /* image height in pixels (MNIST 28x28 padded to 32) */
#define INPUT_W       32     /* image width in pixels */
#define INPUT_C       1      /* input channels: 1 = greyscale, 3 = RGB */
#define NUM_CLASSES   10     /* output classes: digits 0-9 */
#define FP_SHIFT   8
#define FP_ONE     (1 << FP_SHIFT)          /* 256 — represents 1.0 in Q8.8 */
#define FP_ROUND   (1 << (FP_SHIFT - 1))    /* 128 — used for rounding before shift */

typedef int16_t  fp_t;   /* Q8.8 fixed-point value (one activation or weight) */
typedef int32_t  acc_t;  /* wider accumulator used during multiply-accumulate */

static inline fp_t fp_clamp(acc_t x) {
    if (x >  32767) return  32767;
    if (x < -32768) return -32768;
    return (fp_t)x;
}


static inline fp_t fp_mul(fp_t a, fp_t b) {
    return (fp_t)(((acc_t)a * b + FP_ROUND) >> FP_SHIFT);
}


static inline fp_t fp_relu(fp_t x) { return x > 0 ? x : 0; }


__attribute__((noinline))
void CAMPAIGN_END(void)
{
    asm volatile("" ::: "memory");
}

#define IDX3(h, w, c, W, C)     ((h)*(W)*(C) + (w)*(C) + (c))
//#define IDX4(n,h,w,c,H,W,C)     ((n)*(H)*(W)*(C)+(h)*(W)*(C)+(w)*(C)+(c))
#define WIDX(oc,ic,kh,kw,IN,KH,KW) ((((oc)*(IN)+(ic))*(KH)+(kh))*(KW)+(kw)) 


/* =========================================================================
 * Layer size defines — match the exported weight array dimensions
 * ========================================================================= */
#define L0_OUT_C  16    /* conv0 output channels */
#define L1_C      16    /* ResBlock 1 channels (in = out = 16) */
#define L2_IN_C   16    /* ResBlock 2 input channels */
#define L2_OUT_C  32    /* ResBlock 2 output channels (doubles after stride-2) */
#define L3_C      32    /* ResBlock 3 channels (in = out = 32) */


static fp_t buf_a[INPUT_H * INPUT_W * L2_OUT_C]; /* primary working buffer */
static fp_t buf_b[INPUT_H * INPUT_W * L2_OUT_C]; /* secondary/scratch buffer */
static fp_t buf_c[INPUT_H * INPUT_W * L2_OUT_C]; /* third buffer — prevents conv-B input/output aliasing */


static void conv2d(
    const fp_t *in,  int H, int W, int in_c,
          fp_t *out, int OH, int OW, int out_c,
    const fp_t *weights,
    const fp_t *bias,
    int kH, int kW, int stride, int pad)
{
    /* Zero the output buffer — bias is added per-channel in the loop below */
    memset(out, 0, (size_t)(OH * OW * out_c) * sizeof(fp_t));

    for (int oh = 0; oh < OH; oh++) {
        for (int ow = 0; ow < OW; ow++) {
            for (int oc = 0; oc < out_c; oc++) {


                acc_t sum = (acc_t)bias[oc] << FP_SHIFT;

                /* Slide the kH x kW kernel over the input */
                for (int kh = 0; kh < kH; kh++) {
                    int ih = oh * stride - pad + kh; /* corresponding input row */
                    if (ih < 0 || ih >= H) continue; /* skip rows outside input (zero-padding) */

                    for (int kw = 0; kw < kW; kw++) {
                        int iw = ow * stride - pad + kw; /* corresponding input col */
                        if (iw < 0 || iw >= W) continue; /* skip cols outside input */

                        for (int ic = 0; ic < in_c; ic++) {
                            fp_t iv = in[IDX3(ih, iw, ic, W, in_c)];   /* input value */
                            //fp_t wv = weights[
                            //    ((kh * kW + kw) * in_c + ic) * out_c + oc]; /* weight */
                            fp_t wv = weights[((kh * kW + kw) * in_c + ic) * out_c + oc];


                            /*
                             * Multiply-accumulate: both iv and wv are Q8.8,
                             * so iv*wv is Q16.16 (32-bit). We accumulate many
                             * of these before right-shifting at the end.
                             */
                            sum += (acc_t)iv * (acc_t)wv;

                        }
                    }
                }
                out[IDX3(oh, ow, oc, OW, out_c)] =
                    fp_clamp((sum + FP_ROUND) >> FP_SHIFT);
            }
        }
    }

}

static void batchnorm_relu(fp_t *x, int numel_per_c, int C,
                            const fp_t *scale, const fp_t *shift)
{
    for (int i = 0; i < numel_per_c; i++) {
        for (int c = 0; c < C; c++) {
            acc_t v = (acc_t)x[i * C + c];

            /*
             * Apply BN linear transform: v = scale[c] * v + shift[c]
             * scale[c] is Q8.8, so multiplying by it gives Q16.16;
             * right-shifting by FP_SHIFT converts back to Q8.8, then
             * add the shift (also Q8.8).
             */
            v = ((v * scale[c]) >> FP_SHIFT) + shift[c];

            /* Apply ReLU in-place */
            x[i * C + c] = fp_relu(fp_clamp(v));
        }
    }
}

static fp_t _ones16[L2_OUT_C];
static fp_t _zeros16[L2_OUT_C];

static void init_bn_identity(void) {
    for (int i = 0; i < L2_OUT_C; i++) {
        _ones16[i]  = FP_ONE;
        _zeros16[i] = 0;
    }
}

static void resblock(fp_t *x, fp_t *tmp,
                     int H, int W, int C,
                     const fp_t *wa, const fp_t *ba,
                     const fp_t *wb, const fp_t *bb)
{

    /*
     * Conv A: x -> tmp
     * Applies 3x3 conv with stride=1, padding=1 (output same spatial size).
     * Followed immediately by BN (identity) + ReLU.
     */
    conv2d(x, H, W, C, tmp, H, W, C, wa, ba, 3, 3, 1, 1);
    batchnorm_relu(tmp, H * W, C, _ones16, _zeros16);

    /*
     * Conv B: tmp -> x_new (using buf_b as scratch)
     * No ReLU here — the skip connection must be added first before
     * activating. This matches the original ResNet paper design.
     */
    fp_t *x_new = buf_c;   /* NOT buf_b: buf_b is tmp here, would alias conv B in==out */
    conv2d(tmp, H, W, C, x_new, H, W, C, wb, bb, 3, 3, 1, 1);

    /*
     * Add skip connection and apply ReLU.
     * x still holds the original block input (the skip).
     * x_new holds Conv B's output.
     * We add them element-wise and ReLU in-place into x.
     */
    for (int i = 0; i < H * W * C; i++)
        x[i] = fp_relu(fp_clamp((acc_t)x_new[i] + x[i]));

}

static void resblock_ds(
    fp_t *x,
    fp_t *tmp,
    int H, int W, int in_c, int out_c,
    const fp_t *wa, const fp_t *ba,
    const fp_t *wb, const fp_t *bb,
    const fp_t *wp, const fp_t *bp)
{

    int OH = H / 2, OW = W / 2; /* output spatial size after stride-2 */

    /*
     * Projection shortcut: 1x1 conv with stride=2.
     * Transforms input from [H][W][in_c] -> [OH][OW][out_c].
     * Stored in the second half of tmp to avoid clobbering Conv A's output.
     * pad=0 because 1x1 kernels don't need padding.
     */


    fp_t *shortcut = tmp + OH * OW * out_c;
    conv2d(x, H, W, in_c, shortcut, OH, OW, out_c, wp, bp, 1, 1, 2, 0);
    

    /*
     * Conv A: stride=2, so spatial dims halve (H->OH, W->OW).
     * Channels expand from in_c to out_c.
     * Output stored in the first half of tmp.
     */

    conv2d(x, H, W, in_c, tmp, OH, OW, out_c, wa, ba, 3, 3, 2, 1);
    batchnorm_relu(tmp, OH * OW, out_c, _ones16, _zeros16);


    /*
     * Conv B: stays at reduced spatial size, same channel count.
     * Uses buf_b as scratch (main2) — no ReLU yet.
     */

    fp_t *main2 = buf_c;   /* NOT buf_b: buf_b holds tmp+shortcut here, would alias */
    conv2d(tmp, OH, OW, out_c, main2, OH, OW, out_c, wb, bb, 3, 3, 1, 1);
    
    /*
     * Add projection shortcut to Conv B output, then ReLU.
     * Result is written back to x (which is buf_a), overwriting the
     * now-consumed input. buf_a now holds the [OH][OW][out_c] output.
     */

    for (int i = 0; i < OH * OW * out_c; i++)
        x[i] = fp_relu(fp_clamp((acc_t)main2[i] + shortcut[i]));

}

static void global_avg_pool(const fp_t *x, int H, int W, int C, fp_t *out)
{
    int n = H * W; /* total spatial positions to average over */
    for (int c = 0; c < C; c++) {
        acc_t sum = 0;
        /*
         * Sum all H*W activations for channel c.
         * Channels-last layout: channel c at position i is at x[i*C + c].
         */
        for (int i = 0; i < n; i++)
            sum += x[i * C + c];

        /*
         * Divide by H*W to get the mean.
         * Note: when H=W=16, n=256=2^8, so this is equivalent to a right-
         * shift by 8 — the compiler may optimise the division accordingly.
         * fp_clamp guards against the (unlikely) case of sum overflowing int16.
         */
        out[c] = fp_clamp(sum / n);
    }
}

static void fc(const fp_t *in, int in_c,
                     fp_t *out, int out_c,
               const fp_t *W, const fp_t *b)
{
    for (int o = 0; o < out_c; o++) {
        /*
         * Initialise accumulator with bias, pre-shifted to Q16.16 scale
         * so it aligns with the Q16.16 products accumulated below.
         */
        acc_t sum = (acc_t)b[o] << FP_SHIFT;

        /* Dot product: sum += in[i] * W[i][o] for all input features */
        for (int i = 0; i < in_c; i++){
            

            sum += (acc_t)in[i] * W[i * out_c + o];
        }
        /* Convert Q16.16 accumulator back to Q8.8 with rounding and clamping */
        out[o] = fp_clamp((sum + FP_ROUND) >> FP_SHIFT);
    }
}

static int argmax(const fp_t *x, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (x[i] > x[best]) best = i;
    return best;
}


/*static void softmax_display(const fp_t *logits, float *probs, int n)
{
    / Find max logit for numerical stability (used if full softmax added) /
    float fmax = (float)logits[0] / 256.0f;
    for (int i = 1; i < n; i++) {
        float v = (float)logits[i] / 256.0f;
        if (v > fmax) fmax = v;
    }

    / Placeholder: fill probs with 1.0 (softmax not yet implemented) /
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = (float)logits[i] / 256.0f - fmax;
        probs[i] = 1.0f;
        sum += probs[i];
        (void)v; / suppress unused-variable warning /
    }

    / For now just convert raw logits to float — good enough for bar chart /
    for (int i = 0; i < n; i++)
        probs[i] = (float)logits[i] / 256.0f;
} */

static void print_results(const fp_t *logits, int predicted)
{
    /* Find the range of logit values to normalise bar lengths */
    fp_t lmin = logits[0], lmax = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (logits[i] < lmin) lmin = logits[i];
        if (logits[i] > lmax) lmax = logits[i];
    }
    fp_t range = lmax - lmin;
    if (range == 0) range = 1; /* avoid division by zero if all logits equal */

#define BAR_WIDTH 20

    for (int i = 0; i < NUM_CLASSES; i++) {

        /* Bar length: linearly scale logit into [0, BAR_WIDTH] */
        int bar_len = (int)(((long)(logits[i] - lmin) * BAR_WIDTH) / range);
        char bar[BAR_WIDTH + 1];
        for (int b = 0; b < BAR_WIDTH; b++)
            bar[b] = (b < bar_len) ? '#' : '-';
        bar[BAR_WIDTH] = '\0';

        /*
         * Convert Q8.8 logit to a decimal string without printf floats.
         * integer_part = upper 8 bits (whole number portion)
         * frac_part    = lower 8 bits scaled to 0-99 (two decimal places)
         */
        int integer_part = logits[i] >> 8;
        int frac_part    = ((logits[i] & 0xFF) * 100) >> 8;
        if (frac_part < 0) frac_part = -frac_part;

        /* '*' marks the predicted class, ' ' for all others */
    }

}


static fp_t g_logits[NUM_CLASSES];

int resnet_infer_full(const uint16_t *image)
{
    init_bn_identity();

    /* Normalise: uint8 [0,255] -> Q8.8 roughly [-1.0, 1.0] */
    for (int i = 0; i < INPUT_H * INPUT_W * INPUT_C; i++)
        buf_a[i] = image[i];

    /* Conv0 + BN + ReLU: 1ch -> 16ch, 32x32 */
    conv2d(buf_a, INPUT_H, INPUT_W, INPUT_C,
           buf_b, INPUT_H, INPUT_W, L0_OUT_C,
           (const fp_t *)w_conv0, b_conv0, 3, 3, 1, 1);

    batchnorm_relu(buf_b, INPUT_H * INPUT_W, L0_OUT_C, _ones16, _zeros16);
    memcpy(buf_a, buf_b, INPUT_H * INPUT_W * L0_OUT_C * sizeof(fp_t));

    /* ResBlock 1: 16ch, 32x32, identity shortcut */
    resblock(buf_a, buf_b, INPUT_H, INPUT_W, L1_C,
             (const fp_t *)w_rb1_a, b_rb1_a,
             (const fp_t *)w_rb1_b, b_rb1_b);

    /* ResBlock 2: 16->32ch, 32x32->16x16, projection shortcut */
    resblock_ds(buf_a, buf_b, INPUT_H, INPUT_W, L2_IN_C, L2_OUT_C,
                (const fp_t *)w_rb2_a,    b_rb2_a,
                (const fp_t *)w_rb2_b,    b_rb2_b,
                (const fp_t *)w_rb2_proj, b_rb2_proj);

    /* ResBlock 3: 32ch, 16x16, identity shortcut */
    resblock(buf_a, buf_b, INPUT_H / 2, INPUT_W / 2, L3_C,
             (const fp_t *)w_rb3_a, b_rb3_a,
             (const fp_t *)w_rb3_b, b_rb3_b);

    /* Global average pool: 16x16x32 -> 32 */
    fp_t gap[L3_C];
    global_avg_pool(buf_a, INPUT_H / 2, INPUT_W / 2, L3_C, gap);


    /* FC: 32 -> 10 logits, stored in global g_logits for print_results() */
    fc(gap, L3_C, g_logits, NUM_CLASSES, (const fp_t *)w_fc, b_fc);


    return argmax(g_logits, NUM_CLASSES);
}



int main(void)
{
#ifdef PICO_BUILD
    stdio_init_all();

    //sleep_ms(2000);
#endif


    /*
     * Optional:
     * Replace with a real MNIST image later
     */
    /*static const uint8_t test_image[
        INPUT_H * INPUT_W * INPUT_C
    ] = {0};*/


    while(1)
    {

      // CAMPAIGN_LOOP();


        int predicted =
            resnet_infer_full(test_image_q88);

        FI_PREDICTION = predicted;

        FI_SUM = 0;

        for(int i = 0; i < NUM_CLASSES; i++)
        {
            FI_SUM += g_logits[i];
        }
        CAMPAIGN_END();

    }


}
