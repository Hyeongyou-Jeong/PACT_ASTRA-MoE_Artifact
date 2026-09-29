#include "stdint.h"
#include "stdio.h"
#include "fp_inverter.h"

fp16 float_to_fp16(float f)
{
    union {
        float f;
        uint32_t u;
    } fconv;
    fconv.f = f;
    uint32_t fbits = fconv.u;

    uint32_t sign = (fbits >> 31) & 0x1;
    int32_t exp = ((fbits >> 23) & 0xFF) - 127;  // fp32ÀÇ unbiased exponent
    uint32_t frac = fbits & 0x7FFFFF;

    uint16_t h;

    if ((fbits & 0x7F800000) == 0x7F800000) {
        if (frac != 0) {
            h = (uint16_t)((sign << 15) | (0x1F << 10) | (frac >> 13));
        } else {
            h = (uint16_t)((sign << 15) | (0x1F << 10));
        }
        return h;
    }

    if (exp > 15) {
        h = (uint16_t)((sign << 15) | (0x1F << 10));
        return h;
    }

    if (exp < -14) {
        if (exp < -24) {
            h = (uint16_t)(sign << 15);
            return h;
        }
        int shift = (-14 - exp);
        uint32_t subnorm = (frac | 0x00800000) >> (shift + 1);
        h = (uint16_t)((sign << 15) | subnorm);
        return h;
    }

    uint16_t new_exp = (uint16_t)(exp + 15);
    uint16_t new_frac = (uint16_t)(frac >> 13);
    h = (uint16_t)((sign << 15) | (new_exp << 10) | new_frac);
    return h;
}
float fp16_to_float(fp16 h)
{

    uint16_t h_exp = h & 0x7C00;
    uint16_t h_sig = h & 0x03FF;
    uint32_t f_sgn = ((uint32_t)h & 0x8000) << 16;

    uint32_t f_exp, f_sig;
    if (h_exp == 0) {
        if (h_sig == 0) {
            f_exp = 0;
            f_sig = 0;
        } else {
            int shift = 0;
            while ((h_sig & 0x0400) == 0) {
                h_sig <<= 1;
                shift++;
            }
            h_sig &= 0x03FF;
            f_exp = (127 - 14 - shift) << 23;
            f_sig = ((uint32_t)h_sig) << 13;
        }
    } else if (h_exp == 0x7C00) {
        f_exp = 0xFF << 23;
        f_sig = ((uint32_t)h_sig) << 13;
    } else {
        uint32_t exp = (h >> 10) & 0x1F;
        f_exp = (exp - 15 + 127) << 23;
        f_sig = ((uint32_t)h_sig) << 13;
    }

    uint32_t f_bits = f_sgn | f_exp | f_sig;
    union {
        uint32_t u;
        float f;
    } conv;
    conv.u = f_bits;
    return conv.f;
}


bf16 float_to_bf16(float f)
{
    union {
        float f;
        uint32_t u;
    } v;
    v.f = f;

    bf16 bf = (bf16)(v.u >> 16);
    return bf;
}

float bf16_to_float(bf16 b)
{
    union {
        float f;
        uint32_t u;
    } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}
