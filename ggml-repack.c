// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-aarch64.h"
#include "ggml-repack.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>

// ggml_tensor_repacking_mode_t tensor_repacking_mode = TENSOR_REPACKING_MODE_GGML;
ggml_tensor_repacking_mode_t tensor_repacking_mode = TENSOR_REPACKING_MODE_NONE;

ggml_tensor_repacking_mode_t ggml_tensor_repacking_mode () {
    return tensor_repacking_mode;
}

void 
ggml_set_tensor_repacking_mode (
    ggml_tensor_repacking_mode_t mode
    ) 
{
    tensor_repacking_mode = mode;
}

/* TEMP */
inline
float
convert_fp16_to_fp32 (
    ggml_fp16_t x
    ) 
{
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(x)));  
}

void
make_q4_0_k_quant (
    block_q4_0_K * out,
    block_q4_0 * in
    )

//
// Convert from eight block_q4_0 quant blocks to one block_q4_0_k quant block. The
// block_q4_0 quant block are interleaved in the block_q4_0_k quant block such that
// they can be loaded and acted on directly without any shuffles.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t offset;
    block_q8_0 qs_in[8];
    block_q8_0_K qs_out;

    const __m256i m4 = _mm256_set1_epi8(0xf);

    //
    // Unpack q4_0 quant blocks into q8_0 quant blocks.
    //

    for (i = 0; i < 8; i += 1) {
        const __m128i tmp1 = _mm_loadu_si128((const __m128i *)in[i].qs);
        const __m128i tmp2 = _mm_srli_epi16(tmp1, 4);
        __m256i qx = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp1), tmp2, 1);
        qx = _mm256_and_si256(m4, qx);
        _mm256_storeu_si256((__m256i *)qs_in[i].qs, qx);
    }

    //
    // Compute the multiplier value and place it directly in the q4_0_K quant block.
    // 
    // Rearrange the quant bytes into lanes of four bytes interleaved into a q8_0_K
    // quant block.
    // 

    for (i = 0; i < 8; i += 1) {
        out->d[i] = convert_fp16_to_fp32((uint16_t)in[i].d);
        offset = i * 4;

        for (j = 0; j < 8; j += 1) {
            qs_out.qs[offset + 0] = qs_in[i].qs[j * 4 + 0];
            qs_out.qs[offset + 1] = qs_in[i].qs[j * 4 + 1];
            qs_out.qs[offset + 2] = qs_in[i].qs[j * 4 + 2];
            qs_out.qs[offset + 3] = qs_in[i].qs[j * 4 + 3];
            offset += 32;
        }
    }

    //
    // Pack the q8_0_k quant block into the q4_0_k quant block.
    //
    // N.B. The packing of the low and high nibbles in the q4_0_k format is performed
    //      64 quant values rather than the 32 quant values of the q4_0 format. This
    //      makes unpacking of the nibble values in the eventual vector dot function
    //      more efficient.
    //

    for (i = 0; i < 4; i += 1) {
        const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&qs_out.qs[i * 64 + 0]);
        __m256i tmp2 = _mm256_loadu_si256((const __m256i *)&qs_out.qs[i * 64 + 32]);
        tmp2 = _mm256_slli_epi16(tmp2, 4);
        tmp2 = _mm256_or_epi32(tmp1, tmp2);
        _mm256_storeu_si256((__m256i *)&out->qs[i * 32], tmp2);
    }
}

void
make_q8_0_k_quant (
    block_q8_0_K * out,
    block_q8_0 * in
    )

//
// Convert from eight block_q8_0 quant blocks to one block_q8_0_k quant block. The
// block_q8_k quant block are interleaved in the block_q8_0_k quant block such that
// they can be loaded and acted on directly without any shuffles.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t offset;

    //
    // Start temp code - number input bytes.
    //

#if 0
    for (i = 0; i < 8; i += 1) {
        for (j = 0; j < 32; j += 1) {
            in[i].qs[j] = (uint8_t)(j + i * 32);
        }
    }
#endif // #if 0

    //
    // End temp code.
    //

    for (i = 0; i < 8; i += 1) {
        out->d[i] = convert_fp16_to_fp32((uint16_t)in[i].d);
        offset = i * 4;

        for (j = 0; j < 8; j += 1) {
            out->qs[offset + 0] = in[i].qs[j * 4 + 0];
            out->qs[offset + 1] = in[i].qs[j * 4 + 1];
            out->qs[offset + 2] = in[i].qs[j * 4 + 2];
            out->qs[offset + 3] = in[i].qs[j * 4 + 3];
            offset += 32;
        }
    }

    //
    // Start temp code - print out converted quant bytes.
    //

#if 0
    fprintf(logfile, "converted q8_0_k quant bytes\n\n");

    for (i = 0; i < 8; i += 1) {
        for (j = 0; j < 32; j += 1) {
            fprintf(logfile, "%3u ", (uint8_t)out->qs[j + i * 32]);
        }

        fprintf(logfile, "\n\n");
    }
#endif // #if 0

    //
    // End temp code.
    //
}

void
xx_vec_dot_q4_0_K_q8_0_K (
    const uint64_t n,
    float * s,
    size_t bs,
    const block_q4_0_K * x,
    size_t bx,
    const block_q8_0_K * y,
    size_t by,
    int nrc
    )
{
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    const uint64_t nb = n / QK_K;

    __m512 acc = _mm512_setzero_ps();
    __m512i zero512 = _mm512_setzero_si512();

    const __m512i offset = _mm512_set1_epi8(8);
    const __m512i m4 = _mm512_set1_epi8(0xf);

    for (uint64_t i = 0; i < nb; ++i) {

        __m512i sumi = _mm512_setzero_si512();

        //
        // Compute combined scale for the an entire quant block.
        //

        const __m256 xd = _mm256_loadu_ps(x[i].d);
        const __m256 yd = _mm256_loadu_ps(y[i].d);
        const __m256 scale = _mm256_mul_ps(xd, yd);
        __m512 d = _mm512_castps256_ps512(scale);
        d = _mm512_insertf32x8(d, scale, 1);

        //
        // Compute the dot product and accumulate.
        //

        for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
            const __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&x[i].qs[j * QK4_0 + 0]);
            const __m256i tmp2 = _mm256_srli_epi16(tmp1, 4);

            __m512i qx = _mm512_inserti32x8(_mm512_castsi256_si512(tmp1), tmp2, 1);
            qx = _mm512_and_si512(m4, qx);
            qx = _mm512_sub_epi8(qx, offset);

            const __mmask64 is_negative_qx = _mm512_cmp_epi8_mask(qx, zero512, _MM_CMPINT_LT);
            const __m512i negated_qx = _mm512_sub_epi8(zero512, qx);
            __m512i ax = _mm512_mask_mov_epi8(qx, is_negative_qx, negated_qx);

            __m512i qy = _mm512_loadu_si512((const __m512i *)&y[i].qs[j * QK8_0 * 2 + 0]);
            const __m512i negated_qy = _mm512_sub_epi8(zero512, qy);
            __m512i sy = _mm512_mask_mov_epi8(qy, is_negative_qx, negated_qy);

            //
            // mul (ax * sy) + 0 directly to epi32
            //
            // N.B. __AVX512VNNI__ and __AVX512VL__ are always defined.
            //
    
            sumi = _mm512_dpbusd_epi32(sumi, ax, sy);
        }

        //
        // Multiply q with scale and accumulate.
        //

        const __m512 q = _mm512_cvtepi32_ps(sumi);
        acc = _mm512_fmadd_ps(d, q, acc);
    }

    const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                     _mm512_extractf32x8_ps(acc, 1));

    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                 _mm256_extractf128_ps(res, 1));

    const __m128 t1 = _mm_hadd_ps(t0, t0);
    *s = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}

void
xx_vec_dot_q8_0_K_q8_0_K (
    const uint64_t n,
    float * s,
    size_t bs,
    const block_q4_0_K * x,
    size_t bx,
    const block_q8_0_K * y,
    size_t by,
    int nrc
    )
{
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    const uint64_t nb = n / QK_K;

    __m512 acc = _mm512_setzero_ps();
    __m512i zero512 = _mm512_setzero_si512();

    for (uint64_t i = 0; i < nb; ++i) {

        __m512i sumi = _mm512_setzero_si512();

        //
        // Compute combined scale for the an entire quant block.
        //

        const __m256 xd = _mm256_loadu_ps(x[i].d);
        const __m256 yd = _mm256_loadu_ps(y[i].d);
        const __m256 scale = _mm256_mul_ps(xd, yd);
        __m512 d = _mm512_castps256_ps512(scale);
        d = _mm512_insertf32x8(d, scale, 1);

        //
        // Compute the dot product and accumulate.
        //

        for (uint64_t j = 0; j < (QK_K / (QK8_0 * 2)); j += 1) {
            __m512i qx = _mm512_loadu_si512((const __m512i *)&x[i].qs[j * QK8_0 * 2 + 0]);
            const __mmask64 is_negative_qx = _mm512_cmp_epi8_mask(qx, zero512, _MM_CMPINT_LT);
            const __m512i negated_qx = _mm512_sub_epi8(zero512, qx);
            __m512i ax = _mm512_mask_mov_epi8(qx, is_negative_qx, negated_qx);

            __m512i qy = _mm512_loadu_si512((const __m512i *)&y[i].qs[j * QK8_0 * 2 + 0]);
            const __m512i negated_qy = _mm512_sub_epi8(zero512, qy);
            __m512i sy = _mm512_mask_mov_epi8(qy, is_negative_qx, negated_qy);

            //
            // mul (ax * sy) + 0 directly to epi32
            //
            // N.B. __AVX512VNNI__ and __AVX512VL__ are always defined.
            //
    
            sumi = _mm512_dpbusd_epi32(sumi, ax, sy);
        }

        //
        // Multiply q with scale and accumulate.
        //

        const __m512 q = _mm512_cvtepi32_ps(sumi);
        acc = _mm512_fmadd_ps(d, q, acc);
    }

    const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                     _mm512_extractf32x8_ps(acc, 1));

    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                 _mm256_extractf128_ps(res, 1));

    const __m128 t1 = _mm_hadd_ps(t0, t0);
    *s = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}

bool ggml_xbox_repack_tensor (
    struct ggml_tensor *tensor, 
    enum ggml_type repack_type, 
    void * src_data, 
    size_t data_size
    )
{
    block_q4_0 * q40x = src_data;
    block_q4_0_K q4kx_tmp;
    block_q4_0_K * q4kx = tensor->data;
    block_q8_0 * q80x = src_data;
    block_q8_0_K q8kx_tmp;
    block_q8_0_K *q8kx = tensor->data;

    GGML_ASSERT(tensor->type != repack_type);
    GGML_ASSERT(sizeof(block_q4_0) == sizeof(bock_q4_0_K));

    if ((data_size % QK_K) != 0) {
        // number of quants is not 0 mod QK_K
        return false;
    }

    size_t blocks_count = data_size / QK_K;

    if (repack_type == GGML_TYPE_Q4_0_K) {
        for (int i = 0; i < blocks_count; i++) {
            make_q4_0_k_quant(&q4kx_tmp, q40x + i * (QK_K / QK4_0));
            memcpy(q4kx + i, &q4kx_tmp, sizeof(block_q4_0_K));
        }
    } else if (repack_type == GGML_TYPE_Q8_0_K) {
        for (int i = 0; i < blocks_count; i++) {
            make_q8_0_k_quant(&q8kx_tmp, q80x + i * (QK_K / QK8_0));
            memcpy(q8kx + i, &q8kx_tmp, sizeof(block_q8_0_K));
        }
    }

    return true;
}

enum ggml_type ggml_repack_tensor (
    struct ggml_tensor *tensor
    ) 
{
    enum ggml_type type = tensor->type;

    GGML_ASSERT(!tensor->is_repacked);
    GGML_ASSERT(type == GGML_TYPE_Q4_0);
    GGML_ASSERT(type == GGML_TYPE_Q4_K);
    GGML_ASSERT(type == GGML_TYPE_Q8_0);

    if (ggml_tensor_repacking_mode() == TENSOR_REPACKING_MODE_NONE) {
        // no repacking requested
        return type;
    }

    enum ggml_type repack_type = type;
    switch (tensor_repacking_mode) {

        case TENSOR_REPACKING_MODE_GGML:
            //
            // repack GGML mode
            //

            if (type == GGML_TYPE_Q4_0) {
                repack_type = GGML_TYPE_Q4_0_8_8;
            } else if (type == GGML_TYPE_Q4_K) {
                repack_type = GGML_TYPE_Q4_K_8_8;
            }

            if (type != repack_type) {
                size_t data_size = ggml_nbytes(tensor);
#if 1
                void *src_data = malloc(data_size);
                if (src_data == NULL) {
                    return type;
                }
                memcpy(src_data, tensor->data, data_size);
#else                
                void *src_data = tensor->data;
#endif
                if (ggml_aarch64_repack_tensor(tensor, repack_type, src_data, data_size)) {
                    // printf("*** repacking tensor GGML mode %s - type %s - size %zd successfully\n", ggml_get_name(tensor), ggml_type_name(type), data_size);
                    tensor->is_repacked = true;
                    type = repack_type;
                }
#if 1
                free(src_data);
#endif
            }

            break;

        case TENSOR_REPACKING_MODE_XBOX:
            //
            // repack Xbox mode
            //

            if (type == GGML_TYPE_Q4_0) {
                repack_type = GGML_TYPE_Q4_0_K;
            } else if (type == GGML_TYPE_Q8_0) {
                printf("Q8_0 data size %zd\n", ggml_nbytes(tensor));
                repack_type = GGML_TYPE_Q8_0_K;
            }

            if (type != repack_type) {
                size_t data_size = ggml_nbytes(tensor);
#if 1
                void *src_data = malloc(data_size);
                if (src_data == NULL) {
                    return type;
                }
                memcpy(src_data, tensor->data, data_size);
#else                
                void *src_data = tensor->data;
#endif

                if (ggml_xbox_repack_tensor(tensor, repack_type, src_data, data_size)) {
                    // printf("*** repacking tensor Xbox mode %s - type %s - size %zd successfully\n", ggml_get_name(tensor), ggml_type_name(type), data_size);
                    tensor->is_repacked = true;
                    type = repack_type;
                }
#if 1
                free(src_data);
#endif
            }

            break;

        default:
            break;
    }

    return type;
}