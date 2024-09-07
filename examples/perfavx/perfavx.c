#pragma warning (disable:4201) // nameless struct/union

#include <windows.h>
#include <stdlib.h>
#include <intrin.h>
#include <immintrin.h>
#include "ggml.h"

typedef struct {
    const char *api_name;
    void *pfn_benchmark;
    void *pfn_api_AVX2;
    void *pfn_api_AVX512;
} benchmark_api;

typedef void (*PFN_benchmark)(int, benchmark_api *);

#define __GEN_ZA_VERSION__ 1

// add line to cause the file to be modified.
//
// Control for turning data logging on and off.
//

#define __LOG_DATA__

//
// Define default vector size.
//

#define VECTOR_SIZE 1024u
//#define VECTOR_SIZE 256u

//
// Define random number filter to limit the range of generated values.
//

#define FLOAT_FILTER (1 << 14)

//
// Define ggml prototypes.
//

typedef void (*PFN_ggml_init_tables)(void);
PFN_ggml_init_tables pfn_ggml_init_tables_AVX2;
PFN_ggml_init_tables pfn_ggml_init_tables_AVX512;

typedef void (*PFN_ggml_time_init)();
PFN_ggml_time_init pfn_ggml_time_init_AVX2;
PFN_ggml_time_init pfn_ggml_time_init_AVX512;

typedef int64_t (*PFN_ggml_time_us)();
PFN_ggml_time_us pfn_ggml_time_us;

#define QK_K 256u
#define QK8_0 32u

#define K_SCALE_SIZE 12u

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        };
        ggml_half2 dm;
    };
} block_q2_K;

C_ASSERT(!(sizeof(block_q2_K) % sizeof(uint32_t)));

typedef void (*PFN_quantize_row_q2_K)(const float *, block_q2_K *, int64_t);
PFN_quantize_row_q2_K pfn_quantize_row_q2_K_AVX2;
PFN_quantize_row_q2_K pfn_quantize_row_q2_K_AVX512;

void
quantize_row_q2_K (
    const float * x,
    block_q2_K * y,
    int64_t k);

typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        };
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;

C_ASSERT(!(sizeof(block_q4_K) % sizeof(uint32_t)));

void
quantize_row_q4_K (
    const float * x,
    block_q4_K * y,
    int64_t k);

typedef void (*PFN_quantize_row_q4_K)(const float *, block_q4_K *, int64_t);
PFN_quantize_row_q4_K pfn_quantize_row_q4_K_AVX2;
PFN_quantize_row_q4_K pfn_quantize_row_q4_K_AVX512;

void
dequantize_row_q4_K (
    const block_q4_K * x,
    float * y,
    int64_t k);

typedef void (*PFN_dequantize_row_q4_K)(const block_q4_K *, float *, int64_t);
PFN_dequantize_row_q4_K pfn_dequantize_row_q4_K_AVX2;
PFN_dequantize_row_q4_K pfn_dequantize_row_q4_K_AVX512;

typedef struct {
    float d;                            // delta
    int8_t qs[QK_K];                    // quants
    int16_t bsums[QK_K/16];             // sum of quants in groups of 16
} block_q8_K;

C_ASSERT(!(sizeof(block_q8_K) % sizeof(uint32_t)));

typedef void (*PFN_quantize_row_q8_K)(const float *, block_q8_K *, int64_t);
PFN_quantize_row_q8_K pfn_quantize_row_q8_K_AVX2;
PFN_quantize_row_q8_K pfn_quantize_row_q8_K_AVX512;

void
quantize_row_q8_K (
    const float * x,
    block_q8_K * y,
    int64_t k);

typedef struct {
    ggml_half d;                        // delta
    int8_t qs[QK8_0];                   // quants
} block_q8_0;

C_ASSERT(!(sizeof(block_q8_0) % sizeof(uint16_t)));

typedef void (*PFN_quantize_row_q8_0)(const float *, block_q8_0 *, int64_t);
PFN_quantize_row_q8_0 pfn_quantize_row_q8_0_AVX2;
PFN_quantize_row_q8_0 pfn_quantize_row_q8_0_AVX512;

void
quantize_row_q8_0 (
    const float * x,
    block_q8_0 * y,
    int64_t k);

typedef void (*PFN_ggml_fp16_to_fp32_row)(const ggml_fp16_t *, float *, const int64_t);
PFN_ggml_fp16_to_fp32_row pfn_ggml_fp16_to_fp32_row_AVX2;
PFN_ggml_fp16_to_fp32_row pfn_ggml_fp16_to_fp32_row_AVX512;

void
ggml_fp16_to_fp32_row (
    const ggml_fp16_t * x,
    float * y,
    const int64_t n);

typedef void (*PFN_ggml_fp32_to_fp16_row)(const float *, ggml_fp16_t *, const int64_t);
PFN_ggml_fp32_to_fp16_row pfn_ggml_fp32_to_fp16_row_AVX2;
PFN_ggml_fp32_to_fp16_row pfn_ggml_fp32_to_fp16_row_AVX512;

void
ggml_fp32_to_fp16_row (
    const float * x,
    ggml_fp16_t * y,
    const int64_t n);

typedef void (*PFN_ggml_vec_add_f32)(const int32_t, float *, const float *, const float *);
PFN_ggml_vec_add_f32 pfn_ggml_vec_add_f32_AVX2;
PFN_ggml_vec_add_f32 pfn_ggml_vec_add_f32_AVX512;

void
ggml_vec_add_f32 (
    const int32_t n,
    float * z,
    const float * x,
    const float * y);

typedef void (*PFN_ggml_vec_add1_f32)(const int32_t, float *, const float *, const float);
PFN_ggml_vec_add1_f32 pfn_ggml_vec_add1_f32_AVX2;
PFN_ggml_vec_add1_f32 pfn_ggml_vec_add1_f32_AVX512;

void
ggml_vec_add1_f32 (
    const int64_t n,
    float * z,
    const float * x,
    const float v);

typedef void (*PFN_ggml_vec_acc_f32)(const int64_t, float *, const float *);
PFN_ggml_vec_acc_f32 pfn_ggml_vec_acc_f32_AVX2;
PFN_ggml_vec_acc_f32 pfn_ggml_vec_acc_f32_AVX512;

void
ggml_vec_acc_f32 (
    const int64_t n,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_acc1_f32)(const int64_t, float *, const float);
PFN_ggml_vec_acc1_f32 pfn_ggml_vec_acc1_f32_AVX2;
PFN_ggml_vec_acc1_f32 pfn_ggml_vec_acc1_f32_AVX512;

void
ggml_vec_acc1_f32 (
    const int64_t n,
    float * y,
    const float v);

typedef void (*PFN_ggml_vec_sub_f32)(const int32_t, float *, const float *, const float *);
PFN_ggml_vec_sub_f32 pfn_ggml_vec_sub_f32_AVX2;
PFN_ggml_vec_sub_f32 pfn_ggml_vec_sub_f32_AVX512;

void
ggml_vec_sub_f32 (
    const int64_t n,
    float * z,
    const float * x,
    const float * y);

typedef void (*PFN_ggml_vec_set_f32)(const int32_t, float *, const float);
PFN_ggml_vec_set_f32 pfn_ggml_vec_set_f32_AVX2;
PFN_ggml_vec_set_f32 pfn_ggml_vec_set_f32_AVX512;

void
ggml_vec_set_f32 (
    const int64_t n,
    float * x,
    const float v);

typedef void (*PFN_ggml_vec_cpy_f32)(const int32_t, float *, const float *);
PFN_ggml_vec_cpy_f32 pfn_ggml_vec_cpy_f32_AVX2;
PFN_ggml_vec_cpy_f32 pfn_ggml_vec_cpy_f32_AVX512;

void
ggml_vec_cpy_f32 (
    const int64_t n,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_neg_f32)(const int32_t, float *, const float *);
PFN_ggml_vec_neg_f32 pfn_ggml_vec_neg_f32_AVX2;
PFN_ggml_vec_neg_f32 pfn_ggml_vec_neg_f32_AVX512;

void
ggml_vec_neg_f32 (
    const int64_t n,
    float * y,
    const float * x);

typedef void (*PFN_ggml_vec_mul_f32)(const int32_t, float *, const float *, const float *);
PFN_ggml_vec_mul_f32 pfn_ggml_vec_mul_f32_AVX2;
PFN_ggml_vec_mul_f32 pfn_ggml_vec_mul_f32_AVX512;

void
ggml_vec_mul_f32 (
    const int32_t n,
    float * z,
    const float * x,
    const float * y);

typedef void (*PFN_ggml_vec_div_f32)(const int32_t, float *, const float *, const float *);
PFN_ggml_vec_div_f32 pfn_ggml_vec_div_f32_AVX2;
PFN_ggml_vec_div_f32 pfn_ggml_vec_div_f32_AVX512;

void
ggml_vec_div_f32 (
    const int32_t n,
    float * z,
    const float * x,
    const float * y);

typedef void (*PFN_ggml_vec_sum_f32)(const int32_t, float *, const float *);
PFN_ggml_vec_sum_f32 pfn_ggml_vec_sum_f32_AVX2;
PFN_ggml_vec_sum_f32 pfn_ggml_vec_sum_f32_AVX512;

void
ggml_vec_sum_f32 (
    const int32_t n,
    float * s,
    const float * x);

typedef void (*PFN_ggml_vec_sumsq_f32)(const int32_t, float *, const float *);
PFN_ggml_vec_sumsq_f32 pfn_ggml_vec_sumsq_f32_AVX2;
PFN_ggml_vec_sumsq_f32 pfn_ggml_vec_sumsq_f32_AVX512;

void
ggml_vec_sumsq_f32 (
    const int32_t n,
    float * s,
    const float * x);

typedef void (*PFN_ggml_vec_max_f32)(const int32_t, float *, const float *);
PFN_ggml_vec_max_f32 pfn_ggml_vec_max_f32_AVX2;
PFN_ggml_vec_max_f32 pfn_ggml_vec_max_f32_AVX512;

void
ggml_vec_max_f32 (
    const int32_t n,
    float * s,
    const float * x);

typedef void (*PFN_ggml_vec_scale_f32)(const int32_t, float * y, const float v);
PFN_ggml_vec_scale_f32 pfn_ggml_vec_scale_f32_AVX2;
PFN_ggml_vec_scale_f32 pfn_ggml_vec_scale_f32_AVX512;

void
ggml_vec_scale_f32 (
    const int64_t n,
    float * y,
    const float v);

typedef void (*PFN_ggml_vec_mad_f16)(const int, ggml_fp16_t *, const ggml_fp16_t *, const float v);
PFN_ggml_vec_mad_f16 pfn_ggml_vec_mad_f16_AVX2;
PFN_ggml_vec_mad_f16 pfn_ggml_vec_mad_f16_AVX512;

void
ggml_vec_mad_f16 (
    const int n,
    ggml_fp16_t * y,
    const ggml_fp16_t * x,
    const float v);

typedef void (*PFN_ggml_vec_mad_f32)(const int32_t, float * y, const float * x, const float v);
PFN_ggml_vec_mad_f32 pfn_ggml_vec_mad_f32_AVX2;
PFN_ggml_vec_mad_f32 pfn_ggml_vec_mad_f32_AVX512;

void
ggml_vec_mad_f32 (
    const int n,
    float * y,
    const float * x,
    const float v);

typedef void (*PFN_ggml_vec_dot_f16)(int, float *, size_t, const ggml_fp16_t *, size_t, const ggml_fp16_t *, size_t, int);

PFN_ggml_vec_dot_f16 pfn_ggml_vec_dot_f16_AVX2;
PFN_ggml_vec_dot_f16 pfn_ggml_vec_dot_f16_AVX512;

void
ggml_vec_dot_f16 (
    int n,
    float * s,
    size_t bs,
    const ggml_fp16_t * x,
    size_t bx,
    const ggml_fp16_t * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_f32)(int, float *, size_t, const float *, size_t, const float *, size_t, int);
PFN_ggml_vec_dot_f32 pfn_ggml_vec_dot_f32_AVX2;
PFN_ggml_vec_dot_f32 pfn_ggml_vec_dot_f32_AVX512;

void
ggml_vec_dot_f32 (
    int n,
    float * s,
    size_t bs,
    const float * x,
    size_t bx,
    const float * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_f16_f32)(const int64_t, float *, size_t, const ggml_fp16_t *, size_t, const float *, size_t, int);
PFN_ggml_vec_dot_f16_f32 pfn_ggml_vec_dot_f16_f32_AVX2;
PFN_ggml_vec_dot_f16_f32 pfn_ggml_vec_dot_f16_f32_AVX512;

void
ggml_vec_dot_f16_f32 (
    const int64_t n,
    float * s,
    size_t bs,
    const ggml_fp16_t * x,
    size_t bx,
    const float * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_q2_K_q8_K)(int, float *, size_t, const block_q2_K *, size_t, const block_q8_K *, size_t, int);
PFN_ggml_vec_dot_q2_K_q8_K pfn_ggml_vec_dot_q2_K_q8_K_AVX2;
PFN_ggml_vec_dot_q2_K_q8_K pfn_ggml_vec_dot_q2_K_q8_K_AVX512;

void
ggml_vec_dot_q2_K_q8_K (
    int n,
    float * s,
    size_t bs,
    const block_q2_K * x,
    size_t bx,
    const block_q8_K * y,
    size_t by,
    int nrc);

typedef void (*PFN_ggml_vec_dot_q8_0_q8_0)(int, float *, size_t, const void *, size_t, const void *, size_t, int);
PFN_ggml_vec_dot_q8_0_q8_0 pfn_ggml_vec_dot_q8_0_q8_0_AVX2;
PFN_ggml_vec_dot_q8_0_q8_0 pfn_ggml_vec_dot_q8_0_q8_0_AVX512;

void
ggml_vec_dot_q8_0_q8_0 (
    int n,
    float * s,
    size_t bs,
    const void * vx,
    size_t bx,
    const void * vy,
    size_t by,
    int nrc);

//
// Declare logfile descriptor.
//

FILE * logfile = NULL;

//
// Declare iteration values.
//
// N.B. These counts should be such that the respecive test runs for about 10ms.
//
// N.B. Set iter_vec_mad_f16 to 1 and iter_repeat to 1 in order to verify the calculation.
//      Most other values will result in exponent saturation.
//

const uint32_t iter_fp16_to_fp32 = 200000; 
const uint32_t iter_fp32_to_fp16 = 200000; 
const uint32_t iter_vec_add_f32 = 200000;
const uint32_t iter_vec_add1_f32 = 200000;
const uint32_t iter_vec_acc_f32 = 200000;
const uint32_t iter_vec_acc1_f32 = 200000;
const uint32_t iter_vec_sub_f32 = 200000;
const uint32_t iter_vec_set_f32 = 200000;
const uint32_t iter_vec_cpy_f32 = 200000;
const uint32_t iter_vec_neg_f32 = 200000;
const uint32_t iter_vec_mul_f32 = 100000;
const uint32_t iter_vec_div_f32 = 100000;
const uint32_t iter_vec_sum_f32 = 200000;
const uint32_t iter_vec_sumsq_f32 = 200000;
const uint32_t iter_vec_max_f32 = 200000;
const uint32_t iter_vec_scale_f32 = 200000;
//const uint32_t iter_vec_mad_f16 = 1;
const uint32_t iter_vec_mad_f16 = 200000;
const uint32_t iter_vec_mad_f32 = 200000;
const uint32_t iter_vec_dot_f16 = 300000;
const uint32_t iter_vec_dot_f32 = 300000;
const uint32_t iter_vec_dot_f16_f32 = 300000;
const uint32_t iter_vec_dot_q2_K_q8_K = 500000;
const uint32_t iter_vec_dot_q8_0_q8_0 = 500000;
const uint32_t iter_dequantize_q4_k = 50000;
const uint32_t iter_quantize_q8_k = 50000;

//
// Declare repeat iteration count. This the number of times a test will be repeated to get the
// best answer.
//

const uint32_t iter_repeat = 20;
//const uint32_t iter_repeat = 1;

//
// Allocate and free page aligned memory.
//

inline
void *
zalloc (
    size_t size
    )

{

    return VirtualAlloc(NULL,
                        size,
                        MEM_COMMIT | MEM_RESERVE,
                        PAGE_READWRITE); 
}

inline
void
zfree (
    void * base
    )
{

    VirtualFree(base, 0, MEM_DECOMMIT | MEM_RELEASE);
    return;
}

inline
ggml_fp16_t
convert_f32_to_f16 (
    float x
    )

{
    return (ggml_fp16_t)_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), 0), 0); 
}

inline
float
convert_f16_to_f32 (
    ggml_fp16_t x
    )

{
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));  
}

void
log_q2_quant_data (
    uint32_t vec_size,
    uint32_t * z
    )

{

#if !defined(__LOG_DATA__)

    UNREFERENCED_PARAMETER(vec_size);
    UNREFERENCED_PARAMETER(z);

#else

    fprintf(logfile, "quantize row q2 data:\n\n");

    const uint32_t count = vec_size / QK_K;

    for (uint32_t i = 0; i < count; i += 1) {
        uint32_t j;

        for (j = 0; j < 4; j += 1) {
            fprintf(logfile, "%08x ", z[j]); 
        }

        fprintf(logfile, "\n");

        uint32_t l = 0;
        for (; j < 20; j += 1) {
            fprintf(logfile, "%08x ", z[j]);
            l += 1;
            if (!(l & 7)) {
                fprintf(logfile, "\n");
            }
        }

        fprintf(logfile, "%04x %04x - ", (uint16_t)z[j], (uint16_t)(z[j] >> 16));
        fprintf(logfile,
                "(d - %5.2f, dmin - %5.2f)\n\n",
                convert_f16_to_f32((uint16_t)z[j]),
                convert_f16_to_f32((uint16_t)(z[j] >> 16)));

        z += (sizeof(block_q2_K) / sizeof(uint32_t));
    }

    fprintf(logfile, "\n");

#endif // !defined(__LOG_DATA__)

    return;
}

void
log_q4_quant_data (
    uint32_t vec_size,
    uint32_t * z
    )

{

#if !defined(__LOG_DATA__)

    UNREFERENCED_PARAMETER(vec_size);
    UNREFERENCED_PARAMETER(z);

#else

    fprintf(logfile, "quantize row q4 data:\n\n");

    const uint32_t count = vec_size / QK_K;

    for (uint32_t i = 0; i < count; i += 1) {
        uint32_t j;

        fprintf(logfile, "%04x %04x - ", (uint16_t)z[0], (uint16_t)(z[0] >> 16));
        fprintf(logfile,
                "(d - %5.2f, dmin - %5.2f)\n",
                convert_f16_to_f32((uint16_t)z[0]),
                convert_f16_to_f32((uint16_t)(z[0] >> 16)));

        for (j = 1; j < 4; j += 1) {
            fprintf(logfile, "%08x ", z[j]); 
        }

        fprintf(logfile, "\n");

        uint32_t l = 0;
        for (; j < 36; j += 1) {
            fprintf(logfile, "%08x ", z[j]);
            l += 1;
            if (!(l & 7)) {
                fprintf(logfile, "\n");
            }
        }

        fprintf(logfile, "\n");

        z += (sizeof(block_q4_K) / sizeof(uint32_t));
    }

    fprintf(logfile, "\n");

#endif // !defined(__LOG_DATA__)

    return;
}

void
log_q8_quant_data (
    uint32_t vec_size,
    uint32_t * z
    )

{

#if !defined(__LOG_DATA__)

    UNREFERENCED_PARAMETER(vec_size);
    UNREFERENCED_PARAMETER(z);

#else

    fprintf(logfile, "quantize row q8 data:\n\n");

    const uint32_t count = vec_size / QK_K;

    for (uint32_t i = 0; i < count; i += 1) {

        fprintf(logfile, "%08x ", z[0]);
        fprintf(logfile, "(d - %7.2f)\n", *((float  *)(&z[0])));  

        uint32_t j;
        uint32_t l = 0;

        for (j = 1; j < 65; j +=1) {
            fprintf(logfile, "%08x ", z[j]);
            l += 1;
            if (!(l & 7)) {
                fprintf(logfile, "\n");
            }
        }

        l = 0;
        for (; j < 73; j += 1) {
            fprintf(logfile, "%04x %04x ", (uint16_t)z[j], (uint16_t)(z[j] >> 16));
            l += 1;
            if (!(l & 7)) {
                fprintf(logfile, "\n");
            }
        }

        fprintf(logfile, "\n");

        z += (sizeof(block_q8_K) / sizeof(uint32_t)); 
    }

    fprintf(logfile, "\n");

#endif // !define(__LOG_DATA__) 

    return;
}

void
log_q8_0_quant_data (
    uint32_t vec_size,
    uint16_t * z
    )

{

#if !defined(__LOG_DATA__)

    UNREFERENCED_PARAMETER(vec_size);
    UNREFERENCED_PARAMETER(z);

#else

    fprintf(logfile, "quantize row q8_0 data:\n\n");

    const uint32_t count = vec_size / QK8_0;

    for (uint32_t i = 0; i < count; i += 1) {
        fprintf(logfile, "%04x\n", z[0]);

        uint32_t j;
        uint32_t l = 0;
        uint32_t * zd = (void *)(z + 1);

        for (j = 0; j < QK8_0 / 4; j += 1) {
            fprintf(logfile, "%08x ", zd[j]);
            l += 1;
            if (!(l & 7)) {
                fprintf(logfile, "\n");
            }
        }

        fprintf(logfile, "\n");

        z += (sizeof(block_q8_0) / sizeof(uint16_t));
    }

    fprintf(logfile, "\n");

#endif // !define(__LOG_DATA__) 

    return;
}

void
log_raw_data (
    uint32_t count,
    uint32_t * z,
    char * msg
    )

{

#if !defined(__LOG_DATA__)

    UNREFERENCED_PARAMETER(count);
    UNREFERENCED_PARAMETER(z);
    UNREFERENCED_PARAMETER(msg);

#else

    fprintf(logfile, "%s\n\n", msg);

    for (uint32_t i = 0; i < count; i += 1) {
        fprintf(logfile, "%08x ", z[i]);
        if (!((i + 1) & 7)) {
            fprintf(logfile, "\n");
        }
    }
 
    fprintf(logfile, "\n");

#endif // !define(__LOG_DATA__)

    return;
}

void
vec_fp16_to_fp32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )
//
// Compute the performance of vector fp16 converted to float.
//
//  y[i] = x[i]
//
{
    uint32_t i;
    uint32_t j;
    ggml_fp16_t * x;
    float * y;

    //
    // Announce perf test.
    //
    fprintf(logfile, "Running ggml_fp16_to_fp32 performance test for AVX2/AVX512\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_fp16_to_fp32_row_AVX2 = (PFN_ggml_fp16_to_fp32_row)pbapi->pfn_api_AVX2;
    pfn_ggml_fp16_to_fp32_row_AVX512 = (PFN_ggml_fp16_to_fp32_row)pbapi->pfn_api_AVX512;

    //
    // Allocate vectors of the required size.
    //
    x = zalloc(vec_size * sizeof(ggml_fp16_t));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }
    //
    // Fill the vector x with random filtered ggml_fp16_t values.
    //
    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_f32_to_f16((float)(rand() % FLOAT_FILTER));
    }
    //
    // Announce perf test.
    //
    fprintf(logfile, "Running ggml_fp16_to_fp32 performance test for AVX2\n\n");
    //
    // Run the test multiple times to get rid of outliers.
    //
    int64_t best_time_AVX2 = MAXLONG64;
    for (j = 0; j < iter_repeat; j += 1) {
        //
        // Compute the time to do iter_fp16_to_fp32 conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp16_to_fp32; i += 1) {
            pfn_ggml_fp16_to_fp32_row_AVX2(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;
        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_fp16_to_fp32,
            best_time_AVX2);
    //
    // log raw data.
    //
    log_raw_data(vec_size, (void *)y, "y vector output:");
    //
    // Announce perf test.
    //
    fprintf(logfile, "Running ggml_fp16_to_fp32 performance test for AVX512\n\n");
    //
    // Run the test multiple times to get rid of outliers.
    //
    int64_t best_time_AVX512 = MAXLONG64;
    for (j = 0; j < iter_repeat; j += 1) {
        //
        // Compute the time to do iter_fp16_to_fp32 conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp16_to_fp32; i += 1) {
            pfn_ggml_fp16_to_fp32_row_AVX512(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;
        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_fp16_to_fp32,
            best_time_AVX512);
 
    //
    // log raw data.
    //
    log_raw_data(vec_size, (void *)y, "y vector output:");
    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //
    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_fp16_to_fp32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);
exit:
    if (x) {
        zfree(x);
    }
    if (y) {
        zfree(y);
    }
    return;
}

void
vec_fp32_to_fp16 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )
//
// Compute the performance of vector fp32 converted to fp16.
//
//  y[i] = x[i]
//
{
    uint32_t i;
    uint32_t j;
    float * x;
    ggml_fp16_t * y;

    //
    // Announce perf test.
    //
    fprintf(logfile, "Running ggml_fp32_to_fp16 performance test for AVX2/AVX512\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_fp32_to_fp16_row_AVX2 = (PFN_ggml_fp32_to_fp16_row)pbapi->pfn_api_AVX2;
    pfn_ggml_fp32_to_fp16_row_AVX512 = (PFN_ggml_fp32_to_fp16_row)pbapi->pfn_api_AVX512;

    //
    // Allocate vectors of the required size.
    //
    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(ggml_fp16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }
    //
    // Fill the vector x with random filtered float values.
    //
    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
    }
    //
    // Announce perf test.
    //
    fprintf(logfile, "Running ggml_fp32_to_fp16 performance test for AVX2\n\n");
    //
    // Run the test multiple times to get rid of outliers.
    //
    int64_t best_time_AVX2 = MAXLONG64;
    for (j = 0; j < iter_repeat; j += 1) {
        //
        // Compute the time to do iter_fp32_to_fp16 conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp32_to_fp16; i += 1) {
            pfn_ggml_fp32_to_fp16_row_AVX2(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;
        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_fp32_to_fp16,
            best_time_AVX2);
    //
    // log raw data.
    //
    log_raw_data(vec_size / 2, (void *)y, "y vector output:");
    //
    // Announce perf test.
    //
    fprintf(logfile, "Running ggml_fp32_to_fp16 performance test for AVX512\n\n");
    //
    // Run the test multiple times to get rid of outliers.
    //
    int64_t best_time_AVX512 = MAXLONG64;
    for (j = 0; j < iter_repeat; j += 1) {
        //
        // Compute the time to do iter_fp32_to_fp16 conversion of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_fp32_to_fp16; i += 1) {
            pfn_ggml_fp32_to_fp16_row_AVX512(x, y, vec_size);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;
        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_fp32_to_fp16,
            best_time_AVX512);
 
    //
    // log raw data.
    //
    log_raw_data(vec_size / 2, (void *)y, "y vector output:");
    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //
    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_fp32_to_fp16: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);
exit:
    if (x) {
        zfree(x);
    }
    if (y) {
        zfree(y);
    }
    return;
}

void
vec_add_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector add f32.
//
//  z[i] = x[i] + y[i]
//

{
    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_add_f32_AVX2 = (PFN_ggml_vec_add_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_add_f32_AVX512 = (PFN_ggml_vec_add_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
        y[i] = (float)(rand() % FLOAT_FILTER); 
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_add_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_add_f32; i += 1) {
            pfn_ggml_vec_add_f32_AVX2(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_add_f32,
            best_time_AVX2);
 
    //
    // Log raw data output.
    //
 
    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_add_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_add_f32; i += 1) {
            pfn_ggml_vec_add_f32_AVX512(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_add_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    // 
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_add_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_add1_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector add1 f32.
//
//  z[i] = x[i] + v
//

{
    uint32_t i;
    uint32_t j;
    float * x;
    float v;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add1_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_add1_f32_AVX2 = (PFN_ggml_vec_add1_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_add1_f32_AVX512 = (PFN_ggml_vec_add1_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill v and the x vector with random filtered values converted to float.
    //

    v = (float)(rand() % FLOAT_FILTER);
    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_add1_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_add1_f32; i += 1) {
            pfn_ggml_vec_add1_f32_AVX2(vec_size, z, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_add1_f32,
            best_time_AVX2);

    //
    // Log raw data output.
    //
 
    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_add1_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_add1_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_add1_f32; i += 1) {
            pfn_ggml_vec_add1_f32_AVX512(vec_size, z, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_add1_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_add1_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_acc_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector acc f32.
//
//  y[i] += x[i]
//
// N.B. This test repeatedly adds x[i] to y[i] which causes a modified y vector to be used on
//      the next iteration where the y vector is again used as input. Although the value of
//      the y vector is changing, the resultant number across za and zo ggml implementations
//      can stil be compared correctly.
//

{
    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_acc_f32_AVX2 = (PFN_ggml_vec_acc_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_acc_f32_AVX512 = (PFN_ggml_vec_acc_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
        y[i] = (float)(rand() % FLOAT_FILTER); 
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_add_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_acc_f32; i += 1) {
            pfn_ggml_vec_acc_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_acc_f32,
            best_time_AVX2);
 
    //
    // Log raw data output.
    //
 
    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_acc_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_acc_f32; i += 1) {
            pfn_ggml_vec_acc_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_acc_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_acc_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_acc1_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector add1 f32.
//
//  y[i] += v
//
// N.B. This test repeatedly adds v to y[i] which causes a modified y vector to be used on
//      the next iteration where the y vector is again used as input. Although the value of
//      the y vector is changing, the resultant number across za and zo ggml implementations
//      can stil be compared correctly.
//

{
    uint32_t i;
    uint32_t j;
    float v;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc1_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_acc1_f32_AVX2 = (PFN_ggml_vec_acc1_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_acc1_f32_AVX512 = (PFN_ggml_vec_acc1_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    y = zalloc(vec_size * sizeof(float));
    if (!y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill v and the x vector with random filtered values converted to float.
    //

    v = (float)(rand() % FLOAT_FILTER);
    for (i = 0; i < vec_size; i += 1) {
        y[i] = (float)(rand() % FLOAT_FILTER);
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_add1_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_acc1_f32; i += 1) {
            pfn_ggml_vec_acc1_f32_AVX2(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_acc1_f32,
            best_time_AVX2);

    //
    // Log raw data output.
    //
 
    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_acc1_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_add1_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_acc1_f32; i += 1) {
            pfn_ggml_vec_acc1_f32_AVX512(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_acc1_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_acc1_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (y) {
        zfree(y);
    }

    return;
}

void
vec_sub_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector add f32.
//
//  z[i] = x[i] - y[i]
//

{
    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sub_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_sub_f32_AVX2 = (PFN_ggml_vec_sub_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_sub_f32_AVX512 = (PFN_ggml_vec_sub_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
        y[i] = (float)(rand() % FLOAT_FILTER); 
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_sub_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sub_f32; i += 1) {
            pfn_ggml_vec_sub_f32_AVX2(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_sub_f32,
            best_time_AVX2);
 
    //
    // Log raw data output.
    //
 
    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sub_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_sub_f32 vector adds.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sub_f32; i += 1) {
            pfn_ggml_vec_add_f32_AVX512(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_sub_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_sub_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_set_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector set f32.
//
//  x[i] = v
//

{

    uint32_t i;
    uint32_t j;
    float v;
    float * x;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_set_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_set_f32_AVX2 = (PFN_ggml_vec_set_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_set_f32_AVX512 = (PFN_ggml_vec_set_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    if (!x) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill v a random filtered values converted to float.
    //

    v = (float)(rand() % FLOAT_FILTER);

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_set_f32 vector sets.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_set_f32; i += 1) {
            pfn_ggml_vec_set_f32_AVX2(vec_size, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_set_f32,
            best_time_AVX2);

    //
    // Log raw data output.
    //
 
    log_raw_data(vec_size, (void *)x, "x vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_set_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_set_f32 vector sets.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_set_f32; i += 1) {
            pfn_ggml_vec_set_f32_AVX512(vec_size, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_set_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)x, "x vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_set_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_cpy_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector copy f32.
//
//  y[i] = x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cpy_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_cpy_f32_AVX2 = (PFN_ggml_vec_cpy_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_cpy_f32_AVX512 = (PFN_ggml_vec_cpy_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_cpy_f32 vector sets.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_cpy_f32; i += 1) {
            pfn_ggml_vec_cpy_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_cpy_f32,
            best_time_AVX2);

    //
    // Log raw data output.
    //
 
    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_cpy_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_cpy_f32 vector sets.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_cpy_f32; i += 1) {
            pfn_ggml_vec_cpy_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_cpy_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);  
    fprintf(logfile, "  vec_cpy_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_neg_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector negate f32.
//
//  y[i] = -x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_neg_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_neg_f32_AVX2 = (PFN_ggml_vec_neg_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_neg_f32_AVX512 = (PFN_ggml_vec_neg_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //
    // N.B. Make every other value negative.
    //

    for (i = 0; i < vec_size; i += 1) {
        int32_t xc = rand() % FLOAT_FILTER;
        if (i & 1) {
            xc = -xc;
        }

        x[i] = (float)(xc);
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_neg_f32 vector sets.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_neg_f32; i += 1) {
            pfn_ggml_vec_neg_f32_AVX2(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_neg_f32,
            best_time_AVX2);

    //
    // Log raw data output.
    //
 
    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_neg_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_neg_f32 vector sets.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_neg_f32; i += 1) {
            pfn_ggml_vec_neg_f32_AVX512(vec_size, y, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }

    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_neg_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_neg_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_mul_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector multiply f32.
//
//  z[i] = x[i] * y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mul_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_mul_f32_AVX2 = (PFN_ggml_vec_mul_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_mul_f32_AVX512 = (PFN_ggml_vec_mul_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
        y[i] = (float)(rand() % FLOAT_FILTER); 
    }

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_mul_f32 vector multiplies.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mul_f32; i += 1) {
            pfn_ggml_vec_mul_f32_AVX2(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_mul_f32,
            best_time_AVX2);

    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mul_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_mul_f32 vector multiplies.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mul_f32; i += 1) {
            pfn_ggml_vec_mul_f32_AVX512(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_mul_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_mul_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_div_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of vector divide f32.
//
//  z[i] = x[i] / y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float * z;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_div_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_div_f32_AVX2 = (PFN_ggml_vec_div_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_div_f32_AVX512 = (PFN_ggml_vec_div_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
        y[i] = (float)(rand() % FLOAT_FILTER); 
    }

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_div_f32 vector divides.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_div_f32; i += 1) {
            pfn_ggml_vec_div_f32_AVX2(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_div_f32,
            best_time_AVX2);

    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_div_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers. 
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_div_f32 vector divides.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_div_f32; i += 1) {
            pfn_ggml_vec_div_f32_AVX512(vec_size, z, x, y);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_div_f32,
            best_time_AVX512);

    log_raw_data(vec_size, (void *)z, "z vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_div_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
vec_sum_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of summation of the product of f32/f32 vector elements.
//
//  sum += x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float sum;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sum_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_sum_f32_AVX2 = (PFN_ggml_vec_sum_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_sum_f32_AVX512 = (PFN_ggml_vec_sum_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vector of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    if (!x) {
        fprintf(logfile, "  failed to allocate vector \n");
        goto exit;
    }

    //
    // Fill the x with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
    }

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_sum_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sum_f32; i += 1) {
            pfn_ggml_vec_sum_f32_AVX2(vec_size, &sum, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_sum_f32,
            best_time_AVX2);
 
    fprintf(logfile, "  vector sum %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sum_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_sum_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sum_f32; i += 1) {
            pfn_ggml_vec_sum_f32_AVX512(vec_size, &sum, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_sum_f32,
            best_time_AVX512);
 
    fprintf(logfile, "  vector sum %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // best_time_AVX2 = max(1, best_time_AVX2);
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_sum_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_sumsq_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of summation of the product of f32/f32 vector elements.
//
//  sum += x[i] * x[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float sum;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sumsq_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_sumsq_f32_AVX2 = (PFN_ggml_vec_sumsq_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_sumsq_f32_AVX512 = (PFN_ggml_vec_sumsq_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vector of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    if (!x) {
        fprintf(logfile, "  failed to allocate vector \n");
        goto exit;
    }

    //
    // Fill the x with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
    }

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_sumsq_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sumsq_f32; i += 1) {
            pfn_ggml_vec_sumsq_f32_AVX2(vec_size, &sum, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_sumsq_f32,
            best_time_AVX2);
 
    fprintf(logfile, "  vector sum %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_sumsq_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_sumsq_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_sumsq_f32; i += 1) {
            pfn_ggml_vec_sumsq_f32_AVX512(vec_size, &sum, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_sumsq_f32,
            best_time_AVX512);
 
    fprintf(logfile, "  vector sum %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_sumsq_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_max_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of maximum value of the f32/f32 vector elements.
//
//  max = max(x[i], max)
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float max;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_max_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_max_f32_AVX2 = (PFN_ggml_vec_max_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_max_f32_AVX512 = (PFN_ggml_vec_max_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vector of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    if (!x) {
        fprintf(logfile, "  failed to allocate vector \n");
        goto exit;
    }

    //
    // Fill the x with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
    }

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_max_f32 to find the maximum of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_max_f32; i += 1) {
            pfn_ggml_vec_max_f32_AVX2(vec_size, &max, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_max_f32,
            best_time_AVX2);
 
    fprintf(logfile, "  vector max %08x\n\n", *(uint32_t *)&max);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_max_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_max_f32 to find the maximum of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_max_f32; i += 1) {
            pfn_ggml_vec_max_f32_AVX512(vec_size, &max, x);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_max_f32,
            best_time_AVX512);
 
    fprintf(logfile, "  vector max %08x\n\n", *(uint32_t *)&max);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_max_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    return;
}

void
vec_scale_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of the product of a vector elements and a scale value.
//
//  y[i] *= v
//
// N.B. This test repeatedly adds y[i] * v to y[i] which causes a modified y vector to be used
//      on the next iteration where the y vector is again used as input. Although the value of
//      the y vector is changing, the resultant number across za and zo ggml implementations
//      can stil be compared correctly.
//

{

    uint32_t i;
    uint32_t j;
    float * y;
    float v;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_scale_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_scale_f32_AVX2 = (PFN_ggml_vec_scale_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_scale_f32_AVX512 = (PFN_ggml_vec_scale_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vector of the specified size.
    //

    y = zalloc(vec_size * sizeof(float));
    if (!y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill v and the y vector with random filtered values converted to float.
    //
    // N.B. the value of v is set to one to avoid overlowing/saturating individual values
    //      of y[i].
    //

    v = 1.0f;
    for (i = 0; i < vec_size; i += 1) {
        y[i] = (float)(rand() % FLOAT_FILTER);
    }

    //
    // Log raw data output.
    //

    log_raw_data(1, (void *)&v, "v value output:\n");
    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_ssale_f32 summation of the product of vector
        // elements and v.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_scale_f32; i += 1) {
            pfn_ggml_vec_scale_f32_AVX2(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_scale_f32,
            best_time_AVX2);

    //
    // Log raw data output.
    //

    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_scale_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_ssale_f32 summation of the product of vector
        // elements and v.
        //

        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_scale_f32; i += 1) {
            pfn_ggml_vec_scale_f32_AVX512(vec_size, y, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_scale_f32,
            best_time_AVX512);
 
    //
    // Log raw data output.
    //

    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_scale_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (y) {
        zfree(y);
    }

    return;
}

void
vec_mad_f16 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of summation of the product of vector elements with v.
//
//  y[i] += x[i]*v
//
// N.B. This test repeatedly adds y[i] + x[i] * v to y[i] which causes a modified y
//      vector to be used on the next iteration where the y vector is again used as
//      input. Although the value of the y vector is changing, the resultant number
//      across za and zo ggml implementations can stil be compared correctly.
//
                                             
{

    uint32_t i;
    uint32_t j;
    ggml_fp16_t * x;
    ggml_fp16_t * y;
    float v;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f16 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_mad_f16_AVX2 = (PFN_ggml_vec_mad_f16)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_mad_f16_AVX512 = (PFN_ggml_vec_mad_f16)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_fp16_t));
    y = zalloc(vec_size * sizeof(ggml_fp16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill v, x and the y vector with random filtered values converted to float.
    //
    // N.B. the value of v is set to one to avoid overlowing/saturating individual values
    //      of y[i].
    //
    // N.B. Limit the range of numbers to avoid saturating the individual values of y[i].
    //

    v = 1.0;
    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_f32_to_f16((float)(rand() % FLOAT_FILTER));
        y[i] = convert_f32_to_f16((float)(rand() % FLOAT_FILTER));
    }

    //
    // Log raw data output.
    //

    log_raw_data(1, (void *)&v, "v value output:");
    log_raw_data(vec_size / 2, (void *)x, "x vector output:");
    log_raw_data(vec_size / 2, (void *)y, "y vector output:");

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_mad_f16 summation of the product of vector
        // elements and v.
        //

        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mad_f16; i += 1) {
            pfn_ggml_vec_mad_f16_AVX2(vec_size, y, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_mad_f16,
            best_time_AVX2);

    //
    // Log raw data output.
    //

    log_raw_data(vec_size / 2, (void *)y, "y vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f16 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_mad_f16 summation of the product of vector
        // elements and v.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mad_f16; i += 1) {
            pfn_ggml_vec_mad_f16_AVX512(vec_size, y, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_mad_f16,
            best_time_AVX512);
 
    //
    // Log raw data output.
    //

    log_raw_data(vec_size / 2, (void *)y, "y vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_mad_f16: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_mad_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of summation of the product of vector elements with v.
//
//  y[i] += x[i]*v
//
// N.B. This test repeatedly adds y[i] + x[i] * v to y[i] which causes a modified y
//      vector to be used on the next iteration where the y vector is again used as
//      input. Although the value of the y vector is changing, the resultant number
//      across za and zo ggml implementations can stil be compared correctly.
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float v;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_mad_f32_AVX2 = (PFN_ggml_vec_mad_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_mad_f32_AVX512 = (PFN_ggml_vec_mad_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vector\n");
        goto exit;
    }

    //
    // Fill v, x and the y vector with random filtered values converted to float.
    //

    v = (float)(rand() % FLOAT_FILTER);
    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
        y[i] = (float)(rand() % FLOAT_FILTER);
    }

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_mad_f32 summation of the product of vector
        // elements and v.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mad_f32; i += 1) {
            pfn_ggml_vec_mad_f32_AVX2(vec_size, y, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_mad_f32,
            best_time_AVX2);

    //
    // Log raw data output.
    //

    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_mad_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_mad_f32 summation of the product of vector
        // elements and v.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_mad_f32; i += 1) {
            pfn_ggml_vec_mad_f32_AVX512(vec_size, y, x, v);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_mad_f32,
            best_time_AVX512);
 
    //
    // Log raw data output.
    //

    log_raw_data(vec_size, (void *)y, "y vector output:");

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  vec_mad_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
dequantize_q4_k (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of dequantize q4 by generating a quantized set of q4 blocks,
// then dequantizing the block.
//

{

    uint32_t i;
    uint32_t j;
    float * x = NULL;
    block_q4_K * y = NULL;
    float * z = NULL;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q4_k performance tests for AVX2/AVX512\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_dequantize_row_q4_K_AVX2 = (PFN_dequantize_row_q4_K)pbapi->pfn_api_AVX2;
    pfn_dequantize_row_q4_K_AVX512 = (PFN_dequantize_row_q4_K)pbapi->pfn_api_AVX512;

    //
    // Allocate float vector and block_q8_k vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK_K) * sizeof(block_q4_K));
    z = zalloc(vec_size * sizeof(float));
    if (!x || !y || !z) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //
    // N.B. Make every other value negative.
    //

    for (i = 0; i < vec_size; i += 1) {
        int32_t xc = rand() % FLOAT_FILTER;
        if (i & 1) {
            xc = -xc;
        }

        x[i] = (float)xc;
    }

    //
    // Log generated raw x data.
    //

    log_raw_data(vec_size, (void *)x, "q4 float x data:");

    //
    // quantize the x vector of floats.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q4_K_AVX512(x, y, vec_size);

    //
    // log q4 quant data.
    //

    log_q4_quant_data(vec_size, (void *)y);

    //
    // dequantize generated q4 quant blocks.
    //

    pfn_dequantize_row_q4_K_AVX2(y, z, vec_size);
    log_raw_data(vec_size, (void *)z, "AVX2 dequantized q4 float z data:");

    pfn_dequantize_row_q4_K_AVX512(y, z, vec_size);
    log_raw_data(vec_size, (void *)z, "AVX512 dequantized q4 float z data:");

    //
    // Run the test multiple times to get rid of outliers.
    //

    fprintf(logfile, "Running dequantize_q4_k performance test for AVX2\n\n");

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_dequantize_q4_k dequantize calls.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q4_k; i += 1) {
            pfn_dequantize_row_q4_K_AVX2(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_dequantize_q4_k,
            best_time_AVX2);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running dequantize_q4_k performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_dequantize_q4_k dequantize calls.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_dequantize_q4_k; i += 1) {
            pfn_dequantize_row_q4_K_AVX512(y, z, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_dequantize_q4_k,
            best_time_AVX512);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  dequantize_q4_k: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (z) {
        zfree(z);
    }

    return;
}

void
quantize_q8_k (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of the quantization of f32 vector elements to q8_k.
//
//  y = quantize_q8_k(x);
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    block_q8_K * y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_k performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_quantize_row_q8_K_AVX2 = (PFN_quantize_row_q8_K)pbapi->pfn_api_AVX2;
    pfn_quantize_row_q8_K_AVX512 = (PFN_quantize_row_q8_K)pbapi->pfn_api_AVX512;

    //
    // Allocate float vector and block_q8_k vector for quants.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc((vec_size / QK_K) * sizeof(block_q8_K));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x vector with random filtered values converted to float.
    //
    // N.B. Make every other value negative.
    //

    for (i = 0; i < vec_size; i += 1) {
        int32_t xc = rand() % FLOAT_FILTER;
        if (i & 1) {
            xc = -xc;
        }

        x[i] = (float)xc;
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    fprintf(logfile, "Running quantize_q8_k performance test for AVX2\n\n");

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f16 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_k; i += 1) {
            pfn_quantize_row_q8_K_AVX2(x, y, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_quantize_q8_k,
            best_time_AVX2);

    log_q8_quant_data(vec_size, (void *)y);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running quantize_q8_k performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f16 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_quantize_q8_k; i += 1) {
            pfn_quantize_row_q8_K_AVX512(x, y, vec_size); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_quantize_q8_k,
            best_time_AVX512);
 
    //
    // Log raw data output.
    //
 
    log_q8_quant_data(vec_size, (void *)y);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2); 
    fprintf(logfile, "  quantize_q8_k: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_f16 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of summation of the product of f16/f16 vector elements.
//
//  sum = x[i] * y[i]
//

{

    uint32_t i;
    uint32_t j;
    ggml_fp16_t * x;
    ggml_fp16_t * y;
    float sum;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_dot_f16_AVX2 = (PFN_ggml_vec_dot_f16)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_dot_f16_AVX512 = (PFN_ggml_vec_dot_f16)pbapi->pfn_api_AVX512;

    //
    // Allocate f16 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_fp16_t));
    y = zalloc(vec_size * sizeof(ggml_fp16_t));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_f32_to_f16((float)(rand() % FLOAT_FILTER));
        y[i] = convert_f32_to_f16((float)(rand() % FLOAT_FILTER)); 
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f16 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f16; i += 1) {
            pfn_ggml_vec_dot_f16_AVX2(vec_size, &sum, 0, x, 0, y, 0, 1); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_f16,
            best_time_AVX2);
 
    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f16 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f16; i += 1) {
            pfn_ggml_vec_dot_f16_AVX512(vec_size, &sum, 0, x, 0, y, 0, 1); 
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_f16,
            best_time_AVX512);
 
    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_dot_f16: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of summation of the product of f32/f32 vector elements.
//
//  sum = x[i] * y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_dot_f32_AVX2 = (PFN_ggml_vec_dot_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_dot_f32_AVX512 = (PFN_ggml_vec_dot_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
        y[i] = (float)(rand() % FLOAT_FILTER); 
    }

    //
    // Run the test multiple times to get rid of outliers
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f32; i += 1) {
            pfn_ggml_vec_dot_f32_AVX2(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_f32,
            best_time_AVX2);
 
    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times for AVX512 to get rid of outliers
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f32; i += 1) {
            pfn_ggml_vec_dot_f32_AVX512(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_f32,
            best_time_AVX512);
 
    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);  
    fprintf(logfile, "  vec_dot_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_f16_f32 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of summation of the product of f16/f32 vector elements.
//
//  sum = x[i] * y[i]
//

{
    uint32_t i;
    uint32_t j;
    ggml_fp16_t * x;
    float * y;
    float sum;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16_f32 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_dot_f16_f32_AVX2 = (PFN_ggml_vec_dot_f16_f32)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_dot_f16_f32_AVX512 = (PFN_ggml_vec_dot_f16_f32)pbapi->pfn_api_AVX512;

    //
    // Allocate f16 and f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(ggml_fp16_t));
    y = zalloc(vec_size * sizeof(float));
    if (!x || !y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vector with random filtered values converted to float.
    //

    for (i = 0; i < vec_size; i += 1) {
        x[i] = convert_f32_to_f16((float)(rand() % FLOAT_FILTER));
        y[i] = (float)(rand() % FLOAT_FILTER); 
    }

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f16_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f16; i += 1) {
            pfn_ggml_vec_dot_f16_f32_AVX2(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_f16_f32,
            best_time_AVX2);
 
    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_f16_f32 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f16_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_f16; i += 1) {
            pfn_ggml_vec_dot_f16_f32_AVX512(vec_size, &sum, 0, x, 0, y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_f16_f32,
            best_time_AVX512);
 
    fprintf(logfile, "  vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_dot_f16_f32: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    return;
}

void
vec_dot_q2_K_q8_K (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of summation of the product of q2/q8 vector elements.
//
//  sum = q2x[i] * q8y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum;
    block_q2_K * q2x;
    block_q8_K * q8y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q2_K_q8_K performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_dot_q2_K_q8_K_AVX2 = (PFN_ggml_vec_dot_q2_K_q8_K)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_dot_q2_K_q8_K_AVX512 = (PFN_ggml_vec_dot_q2_K_q8_K)pbapi->pfn_api_AVX512;

    //
    // Allocate f16 and f32 vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q2x = zalloc(vec_size / QK_K * sizeof(block_q2_K));
    q8y = zalloc(vec_size / QK_K * sizeof(block_q8_K));
    if (!x || !y || !q2x || !q8y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //
    // N.B. Make every other one negative.

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % 256);
        y[i] = (float)(rand() % FLOAT_FILTER);
        if (i & 1) {
            y[i] = -y[i];
        }
    }

    //
    // Log generated raw x and y data.
    //

    log_raw_data(vec_size, (void *)x, "q2 float x data:");
    log_raw_data(vec_size, (void *)y, "q8 float y data:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q2_K_AVX512(x, q2x, vec_size);
    pfn_quantize_row_q8_K_AVX512(y, q8y, vec_size);

    //
    // log q2 and q8 quant data.
    //

    log_q2_quant_data(vec_size, (void *)q2x);
    log_q8_quant_data(vec_size, (void *)q8y);

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_q2_K_q8_K summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q2_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q2_K_q8_K_AVX2(vec_size, &sum, 0, q2x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_q2_K_q8_K,
            best_time_AVX2);
 
    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q2_K_q8_K performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_f16_f32 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q2_K_q8_K; i += 1) {
            pfn_ggml_vec_dot_q2_K_q8_K_AVX512(vec_size, &sum, 0, q2x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_q2_K_q8_K,
            best_time_AVX512);
 
    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_dot_q2_K_q8_K: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q2x) {
        zfree(q2x);
    }

    if (q8y) {
        zfree(q8y);
    }
}

void
vec_dot_q8_0_q8_0 (
    uint32_t vec_size,
    benchmark_api *pbapi
    )

//
// Compute the performance of the summation of the product of q8_0/q8_0 vector elements.
//
//  sum = q2x[i] * q8y[i]
//

{

    uint32_t i;
    uint32_t j;
    float * x;
    float * y;
    float sum;
    block_q8_0 * q8x;
    block_q8_0 * q8y;

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q8_0_q8_0 performance test for AVX2\n\n");

    //
    // Extract the API for benchmark
    //

    pfn_ggml_vec_dot_q8_0_q8_0_AVX2 = (PFN_ggml_vec_dot_q8_0_q8_0)pbapi->pfn_api_AVX2;
    pfn_ggml_vec_dot_q8_0_q8_0_AVX512 = (PFN_ggml_vec_dot_q8_0_q8_0)pbapi->pfn_api_AVX512;

    //
    // Allocate vectors of the specified size.
    //

    x = zalloc(vec_size * sizeof(float));
    y = zalloc(vec_size * sizeof(float));
    q8x = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    q8y = zalloc(vec_size / QK8_0 * sizeof(block_q8_0));
    if (!x || !y || !q8x || !q8y) {
        fprintf(logfile, "  failed to allocate vectors \n");
        goto exit;
    }

    //
    // Fill the x and y vectors with random filtered values converted to float.
    //
    // N.B. Make every other one negative.

    for (i = 0; i < vec_size; i += 1) {
        x[i] = (float)(rand() % FLOAT_FILTER);
        y[i] = (float)(rand() % FLOAT_FILTER);
        if (i & 1) {
            y[i] = -y[i];

        } else {
            x[i] = -x[i];
        }
    }

    //
    // Log generated raw x and y data.
    //

    log_raw_data(vec_size, (void *)x, "q8_0 float x data:");
    log_raw_data(vec_size, (void *)y, "q8_0 float y data:");

    //
    // quantize the x and y vectors of float.
    //
    // N.B. The quantization routine for both AVX2 and AVX512 is identically the same.
    //

    pfn_quantize_row_q8_0_AVX512(x, q8x, vec_size);
    pfn_quantize_row_q8_0_AVX512(y, q8y, vec_size);

    //
    // log q8 quant data.
    //

    log_q8_0_quant_data(vec_size, (void *)q8x);
    log_q8_0_quant_data(vec_size, (void *)q8y);

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX2 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_q8_0_q8_0 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q8_0_q8_0; i += 1) {
            pfn_ggml_vec_dot_q8_0_q8_0_AVX2(vec_size, &sum, 0, q8x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX2) {
            best_time_AVX2 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_q8_0_q8_0,
            best_time_AVX2);
 
    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Announce perf test.
    //

    fprintf(logfile, "Running vec_dot_q8_0_q8_0 performance test for AVX512\n\n");

    //
    // Run the test multiple times to get rid of outliers.
    //

    int64_t best_time_AVX512 = MAXLONG64;

    for (j = 0; j < iter_repeat; j += 1) {

        //
        // Compute the time to do iter_vec_dot_q8_0_q8_0 summation of the product of vector elements.
        //
    
        const int64_t start_time = pfn_ggml_time_us();
        for (i = 0; i < iter_vec_dot_q8_0_q8_0; i += 1) {
            pfn_ggml_vec_dot_q8_0_q8_0_AVX512(vec_size, &sum, 0, q8x, 0, q8y, 0, 1);
        }
    
        const int64_t total_time = pfn_ggml_time_us() - start_time;

        if (total_time < best_time_AVX512) {
            best_time_AVX512 = total_time;
        }
    }
 
    //
    // Report the total time in microseconds.
    //
 
    fprintf(logfile, "  total time for %d iterations %zdus\n\n",
            iter_vec_dot_q8_0_q8_0,
            best_time_AVX512);
 
    fprintf(logfile, "vector sum of products %08x\n\n", *(uint32_t *)&sum);

    //
    // Report percent increase from AVX2 to AVX512.
    //
    // N.B. Avoid division by zero.
    //

    best_time_AVX2 = max(1, best_time_AVX2);
    fprintf(logfile, "  vec_dot_q8_0_q8_0: percent increase AVX2->AVX512 %zd%%\n\n",
            (best_time_AVX2 - best_time_AVX512) * 100 / best_time_AVX2);

exit:
    if (x) {
        zfree(x);
    }

    if (y) {
        zfree(y);
    }

    if (q8x) {
        zfree(q8x);
    }

    if (q8y) {
        zfree(q8y);
    }
}

int
main (
    int argc,
    char *argv[]
    )
{
    benchmark_api apis[] = {
        { "ggml_fp16_to_fp32_row", (void *) vec_fp16_to_fp32, NULL, NULL },
        { "ggml_fp32_to_fp16_row", (void *) vec_fp32_to_fp16, NULL, NULL },
        { "ggml_vec_add_f32", (void *) vec_add_f32, NULL, NULL },
        { "ggml_vec_add1_f32", (void *) vec_add1_f32, NULL, NULL },
        { "ggml_vec_acc_f32", (void *) vec_acc_f32, NULL, NULL },
        { "ggml_vec_acc1_f32", (void *) vec_acc1_f32, NULL, NULL },
        { "ggml_vec_sub_f32", (void *) vec_sub_f32, NULL, NULL },
        { "ggml_vec_set_f32", (void *) vec_set_f32, NULL, NULL },
        { "ggml_vec_cpy_f32", (void *) vec_cpy_f32, NULL, NULL },
        { "ggml_vec_neg_f32", (void *) vec_neg_f32, NULL, NULL },
        { "ggml_vec_mul_f32", (void *) vec_mul_f32, NULL, NULL },
        { "ggml_vec_div_f32", (void *) vec_div_f32, NULL, NULL },
        { "ggml_vec_sum_f32", (void *) vec_sum_f32, NULL, NULL },
        { "ggml_vec_sumsq_f32", (void *) vec_sumsq_f32, NULL, NULL },
        { "ggml_vec_max_f32", (void *) vec_max_f32, NULL, NULL },
        { "ggml_vec_scale_f32", (void *) vec_scale_f32, NULL, NULL },
        { "ggml_vec_mad_f16", (void *) vec_mad_f16, NULL, NULL },
        { "ggml_vec_mad_f32", (void *) vec_mad_f32, NULL, NULL },
        { "dequantize_row_q4_K", (void *) dequantize_q4_k, NULL, NULL },
        { "quantize_row_q8_K", (void *) quantize_q8_k, NULL, NULL },
        { "ggml_vec_dot_f16", (void *) vec_dot_f16, NULL, NULL },
        { "ggml_vec_dot_f32", (void *) vec_dot_f32, NULL, NULL },
        { "ggml_vec_dot_f16_f32", (void *) vec_dot_f16_f32, NULL, NULL },
        { "ggml_vec_dot_q2_K_q8_K", (void *) vec_dot_q2_K_q8_K, NULL, NULL },
        { "ggml_vec_dot_q8_0_q8_0", (void *) vec_dot_q8_0_q8_0, NULL, NULL },
    };

 int rc = -1;

    //
    // Initialize function pointers
    //

#ifdef __GEN_ZA_VERSION__

    HMODULE dll_AVX2 = LoadLibraryA("za-ggml-avx2.dll");
    if (dll_AVX2 == NULL) {
        printf("failed to load library for 'za-ggml-avx2.dll'\n");
        goto exit;
    }

    HMODULE dll_AVX512 = LoadLibraryA("za-ggml-avx512.dll");
    if (dll_AVX512 == NULL) {
        printf("failed to load library for 'za-ggml-avx512.dll'\n");
        goto exit;
    }

#else

    HMODULE dll_AVX2 = LoadLibraryA("zo-ggml-avx2.dll");
    if (dll_AVX2 == NULL) {
        printf("failed to load library for 'zo-ggml-avx2.dll'\n");
        goto exit;
    }

    HMODULE dll_AVX512 = LoadLibraryA("zo-ggml-avx512.dll");
    if (dll_AVX512 == NULL) {
        printf("failed to load library for 'zo-ggml-avx512.dll'\n");
        goto exit;
    }

#endif // __GEN_ZA_VERSION__

    pfn_ggml_init_tables_AVX2 = (PFN_ggml_init_tables)(GetProcAddress(dll_AVX2, "ggml_init_tables"));
    if (!pfn_ggml_init_tables_AVX2) {
        printf("failed to get proc address of pfn_ggml_init_tables_AVX2\n");
        goto exit;
    }

    pfn_ggml_init_tables_AVX512 = (PFN_ggml_init_tables)(GetProcAddress(dll_AVX512, "ggml_init_tables"));
    if (!pfn_ggml_init_tables_AVX512) {
        printf("failed to get proc address of pfn_ggml_init_tables_AVX512\n");
        goto exit;
    }

    pfn_ggml_time_init_AVX2 = (PFN_ggml_time_init)(GetProcAddress(dll_AVX2, "ggml_time_init"));
    if (!pfn_ggml_time_init_AVX2) {
        printf("failed to get proc address of pfn_ggml_init_time_AVX2\n");
        goto exit;
    }

    pfn_ggml_time_init_AVX512 = (PFN_ggml_time_init)(GetProcAddress(dll_AVX512, "ggml_time_init"));
    if (!pfn_ggml_time_init_AVX512) {
        printf("failed to get proc address of pfn_ggml_init_tables_AVX512\n");
        goto exit;
    }

    // Explicitly called from vec_dot_q2_K_q8_K()
    pfn_quantize_row_q2_K_AVX512 = (PFN_quantize_row_q2_K)(GetProcAddress(dll_AVX512, "quantize_row_q2_K"));
    if (!pfn_quantize_row_q2_K_AVX512) {
        printf("failed to get proc address of pfn_quantize_row_q2_K_AVX512\n");
        goto exit;
    }

    // Explicitly called from dequantize_q4_k()
    pfn_quantize_row_q4_K_AVX512 = (PFN_quantize_row_q4_K)(GetProcAddress(dll_AVX512, "quantize_row_q4_K"));
    if (!pfn_quantize_row_q4_K_AVX512) {
        printf("failed to get proc address of pfn_quantize_row_q4_K_AVX512\n");
        goto exit;
    }

    // Explicitly needed for vec_dot_q8_0_q8_0()
    pfn_quantize_row_q8_0_AVX512 = (PFN_quantize_row_q8_0)(GetProcAddress(dll_AVX512, "quantize_row_q8_0"));
    if (!pfn_quantize_row_q8_0_AVX512) {
        printf("failed to get proc address of pfn_quantize_row_q8_0_AVX512\n");
        goto exit;
    }

    pfn_ggml_time_us = (PFN_ggml_time_us)(GetProcAddress(dll_AVX2, "ggml_time_us"));

    int total_apis = sizeof(apis) / sizeof(benchmark_api);
    bool missing_api = false;
    printf("Loading AVX2/AVX512 entry points\n");
    for (int i = 0; i < total_apis; i++) {
        apis[i].pfn_api_AVX2 = (void *) GetProcAddress(dll_AVX2, apis[i].api_name);
        if (apis[i].pfn_api_AVX2 == NULL) {
            printf("AVX2   API %s could not be located\n", apis[i].api_name);
            missing_api = true;
        } else {
            printf("AVX2   API %s = %I64X\n", apis[i].api_name, (unsigned __int64) apis[i].pfn_api_AVX2);
        }

        apis[i].pfn_api_AVX512 = (void *) GetProcAddress(dll_AVX512, apis[i].api_name);
        if (apis[i].pfn_api_AVX512 == NULL) {
            printf("AVX512 API %s could not be located\n", apis[i].api_name);
            missing_api = true;
        } else {
            printf("AVX512 API %s = %I64X\n", apis[i].api_name, (unsigned __int64) apis[i].pfn_api_AVX512);
        }
    }

    if (missing_api) {
        goto exit;
    }

    //
    // Set default log file name.
    //

#ifdef __GEN_ZA_VERSION__

    char * filename = "perfavxza.log";

#else

    char * filename = "perfavxzo.log";

#endif // __GEN_ZA_VERSION__

    //
    // Check is a log file is specified.
    //

    if (argc > 1) {
        filename = argv[1];
    }

    //
    // Open log file for write.
    //

    printf("log filename is %s\n", filename);

    logfile = fopen(filename, "w");
    if (!logfile) {
        printf("failed to open log file %s\n", argv[1]);
        goto exit;
    }

    //
    // Set thread affinity - ignore if failure.
    //

    uint64_t affinity;

    if (SetThreadAffinityMask(GetCurrentThread(), 1ull << 8)) {
        affinity = SetThreadAffinityMask(GetCurrentThread(), 1ull << 8);
        fprintf(logfile, "running with group afffinity %p\n", (void *)affinity);

    } else {
        printf("failed to set thread affinity ignored\n");
    }

    //
    // Set thread priority - ignore if failure.
    //

    if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL)) {
        fprintf(logfile, "running at time critical priority\n");

    } else {
        printf("failed to set thread priority ignored\n");
    }

    fprintf(logfile, "\n");

    //
    // Initialize ggml floating conversion tables.
    //
    // N.B. The timing code must be explicitly initialized for both libraries. The initialization
    //      of the ggml tables depends on this. 
    //

    pfn_ggml_time_init_AVX2();
    pfn_ggml_init_tables_AVX2();

    pfn_ggml_time_init_AVX512();
    pfn_ggml_init_tables_AVX512(); 

    //
    // Set the random number seed to always start with the same value so results are the same
    // between runs and between avx2 and avx512.
    //

    srand(47);

    //
    // Compute the performance of various avx code paths.
    //

    for (int i = 0; i < total_apis; i++) {
        if ((apis[i].pfn_api_AVX2 != NULL) && (apis[i].pfn_api_AVX2 != NULL)) {
            ((PFN_benchmark) (apis[i].pfn_benchmark))(VECTOR_SIZE, &apis[i]);
        }
    }

    rc = 0;

exit:
    if (logfile) {
        fclose(logfile);
    }

    return rc;
}
