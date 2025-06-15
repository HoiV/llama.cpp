// SPDX-FileCopyrightText: Copyright 2024 Arm Ltd.
#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

//
// for tensor repacking
//

typedef enum {
    TENSOR_REPACKING_MODE_NONE = 0,
    TENSOR_REPACKING_MODE_GGML = 1,
    TENSOR_REPACKING_MODE_XBOX = 2,
    TENSOR_REPACKING_MODE_MAX  = 3
} ggml_tensor_repacking_mode_t;

typedef struct {
    ggml_half d[8];                     // delta (scale)
    int8_t qs[QK_K / 2];                // quants interleaved packed two per byte
} block_q4_0_repack;

static_assert((sizeof(block_q4_0) * 8) == sizeof(block_q4_0_repack));

typedef struct {
    ggml_half d[8];                     // delta (scale)
    int8_t qs[QK_K];                    // quants interleaved
} block_q8_0_repack;

static_assert((sizeof(block_q8_0) * 8) == sizeof(block_q8_0_repack));

typedef block_q4_K block_q4_K_repack;
typedef block_q8_K block_q8_K_repack;

ggml_tensor_repacking_mode_t ggml_tensor_repacking_scheme(ggml_tensor_repacking_mode_t t);

void ggml_set_tensor_repacking_mode(ggml_tensor_repacking_mode_t type);

enum ggml_type ggml_repack_tensor(struct ggml_tensor *tensor);

// vec_dot routines for Xbox repacked tensors

void
xx_vec_dot_q4_0_q8_0_x8 (
    const int n,
    float * s,
    size_t bs,
    const block_q4_0_repack * x,
    size_t bx,
    const block_q8_0_repack * y,
    size_t by,
    int nrc);

void
xx_vec_dot_q4_k_q8_k_x8 (
    const int n,
    float * s,
    size_t bs,
    const block_q4_K_repack * x,
    size_t bx,
    const block_q8_K_repack * y,
    size_t by,
    int nrc);

void
xx_vec_dot_q8_0_q8_0_x8 (
    const int n,
    float * s,
    size_t bs,
    const block_q8_0_repack * x,
    size_t bx,
    const block_q8_0_repack * y,
    size_t by,
    int nrc);

void                   
quantize_row_q4_k_x8 (
    const float * x,
    block_q4_K * y,
    uint64_t vec_size);

void                   
quantize_row_q8_k_x8 (
    const float * x,
    block_q8_K * y,
    uint64_t vec_size);

void
quantize_row_q8_0_x8 (
    const float * x,
    block_q8_0 * y,
    uint64_t vec_size);

#ifdef __cplusplus
}
#endif
