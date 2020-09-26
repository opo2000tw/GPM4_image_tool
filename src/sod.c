#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "../inc/sod.h"
#include "../inc/def.h"

/*
 * CAPIREF: Refer to the official documentation at https://sod.pixlab.io/api.html for the expected parameters this interface takes.
 */
sod_img sod_make_image(int w, int h, int c)
{
    sod_img out = sod_make_empty_image(w, h, c);
    out.data = calloc(h * w * c, sizeof(float));
    return out;
}

static inline float vy1uy0_get_pixel(sod_img m, int x, int y, int c)
{
    return (m.data ? m.data[c * m.h * m.w + y * m.w + x] : 0.0f);
}
static inline void vy1uy0_set_pixel(sod_img m, int x, int y, int c, float val)
{
    /* x, y, c are already validated by upper layers */
    if (m.data)
        m.data[c * m.h * m.w + y * m.w + x] = val;
}
void sod_img_yuv422_vy1uy0_to_gray(sod_img im)
{
    int i, j;
    float r, g, b;
    float y, u, v;
    if (im.h % 4 != 0 || im.w % 4 != 0)
    {
        return;
    }
    if (im.c != 3 || im.h % 4 || )
    {
        return;
    }
    for (j = 0; j < im.h / 4; ++j)
    {
        for (i = 0; i < im.w / 4; ++i)
        {
            for (size_t i = 0; i < count; i++)
            {
                /* code */
            }
        }
    }
}

/*
 * CAPIREF: Refer to the official documentation at https://sod.pixlab.io/api.html for the expected parameters this interface takes.
 */
#define OTSU_GRAYLEVEL 256
sod_img sod_otsu_binarize_image(sod_img im)
{
    sod_img t = sod_make_image(im.w, im.h, im.c);
    if (t.data)
    {
        /* binarization by Otsu's method based on maximization of inter-class variance */
        int hist[OTSU_GRAYLEVEL];
        double prob[OTSU_GRAYLEVEL], omega[OTSU_GRAYLEVEL]; /* prob of graylevels */
        double myu[OTSU_GRAYLEVEL];   /* mean value for separation */
        double max_sigma, sigma[OTSU_GRAYLEVEL]; /* inter-class variance */
        float threshold; /* threshold for binarization */
        int i; /* Loop variable */
        /* Histogram generation */
        for (i = 0; i < OTSU_GRAYLEVEL; i++) hist[i] = 0;
        for (i = 0; i < im.w * im.h * im.c; ++i)
        {
            hist[(unsigned char)(255. * im.data[i])]++;
        }
        /* calculation of probability density */
        for (i = 0; i < OTSU_GRAYLEVEL; i++)
        {
            prob[i] = (double)hist[i] / (im.w * im.h);
        }
        omega[0] = prob[0];
        myu[0] = 0.0;       /* 0.0 times prob[0] equals zero */
        for (i = 1; i < OTSU_GRAYLEVEL; i++)
        {
            omega[i] = omega[i - 1] + prob[i];
            myu[i] = myu[i - 1] + i * prob[i];
        }
        /* sigma maximization
        sigma stands for inter-class variance
        and determines optimal threshold value */
        threshold = 0.0;
        max_sigma = 0.0;
        for (i = 0; i < OTSU_GRAYLEVEL - 1; i++)
        {
            if (omega[i] != 0.0 && omega[i] != 1.0)
                sigma[i] = pow(myu[OTSU_GRAYLEVEL - 1] * omega[i] - myu[i], 2) /
                           (omega[i] * (1.0 - omega[i]));
            else
                sigma[i] = 0.0;
            if (sigma[i] > max_sigma)
            {
                max_sigma = sigma[i];
                threshold = (float)i;
            }
        }
        threshold /= 255.;
        /* binarization output */
        for (i = 0; i < im.w * im.h * im.c; ++i)
        {
            t.data[i] = im.data[i] > threshold ? 1 : 0;
        }
    }
    return t;
}