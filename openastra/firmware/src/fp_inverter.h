#ifndef __FP_INVERTER_H
#define __FP_INVERTER_H

#include <stdint.h>

typedef uint16_t fp16;
typedef uint16_t bf16;

// 함수 선언만
fp16 float_to_fp16(float f);
float fp16_to_float(fp16 h);

bf16 float_to_bf16(float f);
float bf16_to_float(bf16 b);


#endif // FP_INVERTER_H
