// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include "ggml-aarch64.h"
#include "ggml-repack.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdio.h>

// ggml_tensor_repacking_mode_t tensor_repacking_mode = TENSOR_REPACKING_MODE_GGML;
ggml_tensor_repacking_mode_t tensor_repacking_mode = TENSOR_REPACKING_MODE_NONE;

inline
ggml_tensor_repacking_mode_t
ggml_tensor_repacking_mode (
    void
    )
{
    return tensor_repacking_mode;
}

void 
ggml_set_tensor_repacking_mode (
    ggml_tensor_repacking_mode_t mode
    ) 
{
    tensor_repacking_mode = mode;
}

void
make_q4_0_repack_quant (
    uint64_t ne,
    block_q4_0_repack * out,
    block_q4_0 * in
    )

//
// Convert groups of eight q4_0 quant blocks to one q4_0_repack quant block. The q4_0
// quant blocks are interleaved in the q4_0_repack quant block such that they can be
// loaded and acted on directly.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    block_q8_0 qs_in[8];
    block_q8_0_repack qs_out;

    const __m256i m4 = _mm256_set1_epi8(0xf);

    //
    // Convert groups of eight q4_0 quant blocks into one q4_0_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {
        for (i = 0; i < 8; i += 1) {

            //
            // Unpack 8 q4_0 quant blocks into 8 temporary q8_0 quant blocks.
            //
            // N.B. The multiplier value from the q4_0 quant blocks is moved
            //      directly to the temporary q8_0_repack quant block.
            //

            qs_out.d[i] = in[i].d;
            const __m128i tmp1 = _mm_loadu_si128((const __m128i *)in[i].qs);
            const __m128i tmp2 = _mm_srli_epi16(tmp1, 4);
            __m256i qx = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp1), tmp2, 1);
            qx = _mm256_and_si256(m4, qx);
            _mm256_storeu_si256((__m256i *)qs_in[i].qs, qx);
        }
    
        //
        // Rearrange the quant bytes into lanes of four bytes interleaved into the
        // temporary q8_0_repack quant block.
        // 
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out.qs[offset + 0];
                uint32_t * qs_src = (uint32_t *)&qs_in[i].qs[j * 4 + 0];
                *qs_dst = *qs_src;
                offset += 32;
            }
        }
    
        //
        // Repack the temporary q8_0_repack quant block into the q4_0_repack quant block.
        //
        // N.B. The packing of the low and high nibbles in the q4_0_repack format is performed
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

        //
        // Copy the multiplier values to the q4_0_repack quant block.
        //

        const __m128i d = _mm_loadu_si128((__m128i *)qs_out.d);
        _mm_storeu_si128((__m128i *)out->d, d);

        out += 1;
        in += 8;
    }
}

void
make_q4_k_repack_quant (
    uint64_t ne,
    block_q4_K_repack * out,
    block_q4_K * in
    )

//
// Convert one q4_K quant block to one q4_K_repack quant block. The q4_K quant block
// values are interleaved in the q4_K_repack quant block such that they can be loaded
// and acted on directly.
//
// N.B. The q4_k quant block and q4_K_repack quant block have the same layout and storage
//      requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_in[QK_K];
    uint8_t qs_out[QK_K];

    //
    // Convert one q4_k quant block to one q4_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multipliers/(d-dmin) and scales/(mins-scales) values directly
        // from the input quant block to the output quant block.
        //
    
        out->dm = in->dm;
        memcpy(out->scales, in->scales, sizeof(in->scales));
    
        //
        // Unpack the 4-bit q4_k quant values into an array of 8-bit quant values.
        //
    
        const __m512i m4 = _mm512_set1_epi8(0xf);
    
        for (i = 0; i < (QK_K / 64); i += 1) {
            __m256i tmp1 = _mm256_loadu_si256((const __m256i *)(in->qs + i * 32));
            __m256i tmp2 = _mm256_srli_epi16(tmp1, 4);
    
            __m512i qx = _mm512_castsi256_si512(tmp1); 
            qx = _mm512_inserti32x8(qx, tmp2, 1);
    
            qx = _mm512_and_si512(m4, qx);
            _mm512_storeu_si512((__m512i *)(&qs_in[i * 64]), qx);
        }
    
        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        // 
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&qs_in[i * 32 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 32;
            }
        }
    
        //
        // Repack the q8_k quant block into the q4_k_repack quant block.
        //
        // N.B. The packing of the low and high nibbles in the q4_k_repack format is
        //      performed as 64 quant values rather than the 32 quant values of the
        //      q4_k format. This makes unpacking of the nibble values in the eventual
        //      vector dot function more efficient.
        //
    
        for (i = 0; i < 4; i += 1) {
            __m256i tmp1 = _mm256_loadu_si256((const __m256i *)&qs_out[i * 64 + 0]);
            __m256i tmp2 = _mm256_loadu_si256((const __m256i *)&qs_out[i * 64 + 32]);
    
            tmp2 = _mm256_slli_epi16(tmp2, 4);
            tmp2 = _mm256_or_epi32(tmp1, tmp2);
            _mm256_storeu_si256((__m256i *)&out->qs[i * 32], tmp2);
        }

        out += 1;
        in += 1;
    }
}

void
make_q8_0_repack_quant (
    uint64_t ne,
    block_q8_0_repack * out,
    block_q8_0 * in
    )

//
// Convert groups of eight q8_0 quant blocks to one q8_0_repack quant block. The q8_0
// quant blocks are interleaved in the q8_0_repack quant block such that they can be
// loaded and acted on directly.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    block_q8_0_repack qs_out;

    //
    // Convert groups of eight q8_0 quant blocks into one q8_0_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {
        for (i = 0; i < 8; i += 1) {
            qs_out.d[i] = in[i].d;
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out.qs[offset + 0];
                uint32_t * qs_src = (uint32_t *)&in[i].qs[j * 4 + 0];
                *qs_dst = *qs_src;
                offset += 32;
            }
        }

        //
        // Copy the temporary q8_0_repack quant block to the output quant block.
        //

        memcpy(out, &qs_out, sizeof(*out));

        out += 1;
        in += 8;
    }
}
    
void
make_q8_k_repack_quant (
    uint64_t ne,
    block_q8_K_repack * out,
    block_q8_K * in
    )

//
// Convert one q8_K quant block to one q8_K_repack quant block. The q8_K quant block
// values are interleaved in the block_q8_K_repack quant block such that they can be
// loaded and acted on directly.
//
// N.B. The block_q8_k quant block and block_q8_K_repack quant block have the same
//      layout and storage requirements.
//

{

    uint64_t i;
    uint64_t j;
    uint64_t k;
    uint64_t offset;
    uint8_t qs_out[QK_K];

    //
    // Convert one q8_k quant block to one q8_k_repack quant block.
    //

    uint64_t nb = ne / QK_K;

    for (k = 0; k < nb; k += 1) {

        //
        // Copy the multiplier/(d) and sums/(bsums) values directly from the input quant
        // block to the output quant block.
        //
    
        out->d = in->d;
        memcpy(out->bsums, in->bsums, sizeof(in->bsums));
    
        //
        // Rearrange the 8-bit quant values into lanes of four bytes interleaved.
        // 
    
        for (i = 0; i < 8; i += 1) {
            offset = i * 4;
    
            for (j = 0; j < 8; j += 1) {
                uint32_t * qs_dst = (uint32_t *)&qs_out[offset];
                uint32_t * qs_src = (uint32_t *)&in->qs[i * 32 + j * 4];
                *qs_dst = *qs_src;
    
                offset += 32;
            }
        }
    
        //
        // Copy the rearranged q8_k quant block into the q8_k_repack quant block.
        //
    
        memcpy(out->qs, qs_out, sizeof(out->qs));

        out += 1;
        in += 1;
    }
}

void
xx_vec_dot_q4_0_q8_0_x8 (
    const int n,
    float * s,
    size_t bs,
    const block_q4_0_repack * x,
    size_t bx,
    const block_q8_0_repack * y,
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

        const __m128h xd = _mm_loadu_ph(x[i].d);
        const __m128h yd = _mm_loadu_ph(y[i].d);
        const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                           _mm256_cvtph_ps(yd));

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
xx_vec_dot_q4_k_q8_k_x8 (
    const int n,
    float * s,
    size_t bs,
    const block_q4_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t by,
    int nrc
    )
{
    GGML_UNUSED(bs);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(nrc);

    const uint64_t nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask4 = 0xc0c0c0c0;

    uint32_t utmp[4];

    __m512 acc = _mm512_setzero_ps();
    const __m512i m4 = _mm512_set1_epi8(0xf);
    const __m512 zero512 = _mm512_setzero_ps();

    for (uint64_t i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint32_t * vscales = (uint32_t *)x[i].scales;
        utmp[3] = ((vscales[2] >> 4) & kmask2) | ((vscales[1] & kmask4) >> 2);
        utmp[2] = vscales[1] & kmask1;
        utmp[1] = (vscales[2] & kmask2) | ((vscales[0] & kmask4) >> 2);
        utmp[0] = vscales[0] & kmask1;

        const uint8_t * q4 = x[i].qs;
        const int8_t  * q8 = y[i].qs;

        //
        // Load q4 mins and scales and expand to 16 - 16-bit values.
        //

        const __m128i mins_and_scales8 = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(mins_and_scales8);

        //
        // Load q8 16 bsums values and reduce to 8 bsums values.
        //

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);

        const __m128i q8s = _mm_hadd_epi16(_mm256_castsi256_si128(q8sums),
                                           _mm256_extracti128_si256(q8sums, 1));

        //
        // Multiply the 8 bsums by the 8 mins values, convert to float and add to the
        // vector dot accumulation.
        //

        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);

        const __m128 prod_m = _mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod));
        acc = _mm512_add_ps(acc, _mm512_insertf32x4(zero512, prod_m, 0));

        //
        // Compute the scale vector.
        //
        // N.B. The 8 scale values and replicated to 16 scale values.

        __m512i scale = _mm512_cvtepu16_epi32(mins_and_scales);
        scale = _mm512_inserti64x4(scale, _mm512_castsi512_si256(scale), 1);

        __m512i sumi = _mm512_setzero_si512();

        for (uint64_t j = 0; j < QK_K / 64; ++j) {
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(q4 + (j * 32)));
            const __m512i q8v = _mm512_loadu_si512(q8 + (j * 64));

            __m512i q4v = _mm512_castsi256_si512(q4bits);
            q4v = _mm512_inserti32x8(q4v, _mm256_srli_epi16(q4bits, 4), 1);
            q4v = _mm512_and_si512(q4v, m4);

            sumi = _mm512_dpbusd_epi32(sumi, q4v, q8v);
        }

        sumi = _mm512_mullo_epi32(sumi, scale);
        acc = _mm512_fmadd_ps(_mm512_set1_ps(d), _mm512_cvtepi32_ps(sumi), acc);
    }

    const __m256 res = _mm256_add_ps(_mm512_castps512_ps256(acc),
                                     _mm512_extractf32x8_ps(acc, 1));

    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(res),
                                 _mm256_extractf128_ps(res, 1));

    const __m128 t1 = _mm_hadd_ps(t0, t0);
    *s = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}

void
xx_vec_dot_q8_0_q8_0_x8 (
    const int n,
    float * s,
    size_t bs,
    const block_q8_0_repack * x,
    size_t bx,
    const block_q8_0_repack * y,
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

        const __m128h xd = _mm_loadu_ph(x[i].d);
        const __m128h yd = _mm_loadu_ph(y[i].d);
        const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(xd),
                                           _mm256_cvtph_ps(yd));

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

void                   
quantize_row_q4_k_x8 (
    const float * x,
    block_q4_K * y,
    uint64_t vec_size
    )
{

    //
    // Quantize the x vector into q4_K quants.
    //

    quantize_row_q4_K(x, y, vec_size);

    //
    // Make q4_k_repack quant blocks.
    //

    make_q4_k_repack_quant(vec_size, (block_q4_K_repack *)y, y);
}

void                   
quantize_row_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint64_t vec_size
    )
{

    //
    // Quantize the x vector into q8_K quants.
    //

    quantize_row_q8_K(x, y, vec_size);

    //
    // Make q8_k_repack quant blocks.
    //

    make_q8_k_repack_quant(vec_size, (block_q8_K_repack *)y, y);
}

void
quantize_row_q8_0_x8 (
    const float * x,
    block_q8_0 * y,
    uint64_t vec_size
    )
{

    //
    // Quantize the x vector into q8_0 guants.
    //

    quantize_row_q8_0(x, y, vec_size);

    //
    // Make q8_0_repack quant blocks.
    //

    make_q8_0_repack_quant(vec_size, (block_q8_0_repack *)y, y);
}

enum ggml_type ggml_repack_tensor (
    struct ggml_tensor *tensor
    ) 
{
    enum ggml_type type = tensor->type;

    GGML_ASSERT(!tensor->is_repacked);
    GGML_ASSERT((type == GGML_TYPE_Q4_0) ||
                (type == GGML_TYPE_Q4_K) ||
                (type == GGML_TYPE_Q8_0));

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
            void *src_data = tensor->data;

            if (ggml_aarch64_repack_tensor(tensor, repack_type, src_data, data_size)) {

                // printf("*** converted tensor %s - type %s - size %zd succeeded\n",
                //        ggml_get_name(tensor),
                //        ggml_type_name(type),
                //        data_size);

                type = repack_type;
            }
        }

        break;

    case TENSOR_REPACKING_MODE_XBOX:

        //
        // Check if the number of elements is 0 mod QK_K.
        //

        uint64_t ne = tensor->ne[0];
        if ((ne % QK_K) != 0) {
            return type;
        }

        //
        // Make transformed quant based on current type.
        //
        // N.B. The original data and the repacked data share the same memory. They are
        //      exactly the same size and layout. The make repack quant function does
        //      the repack such that no extra memory needs to be allocated and there are
        //      no extra copies.
        //

        void * src_data = tensor->data;
        if (type == GGML_TYPE_Q4_0) {
            repack_type = GGML_TYPE_Q4_0_x8;
            make_q4_0_repack_quant(ne, src_data, src_data);

        } else if (type == GGML_TYPE_Q4_K) {
            repack_type = GGML_TYPE_Q4_K_x8;
            make_q4_k_repack_quant(ne, src_data, src_data);

        } else if (type == GGML_TYPE_Q8_0) {
            repack_type = GGML_TYPE_Q8_0_Q8_0_x8;
            make_q8_0_repack_quant(ne, src_data, src_data);
        }

        if (type != repack_type) {

            // printf("*** XBOX convert tensor %s - type %s - elements %zd succeeded\n",
            //        ggml_get_name(tensor),
            //        ggml_type_name(type),
            //        tensor->ne[0]);

            type = repack_type;
        }

        break;

    case TENSOR_REPACKING_MODE_NONE:
    default:
        break;
    }

    return type;
}
