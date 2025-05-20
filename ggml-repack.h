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
    float d[8];                         // delta (scale)
    int8_t qs[QK_K / 2];                // quants interleaved packed two per byte
} block_q4_0_K;

typedef struct {
    float d[8];                         // delta (scale)
    int8_t qs[QK_K];                    // quants interleaved
} block_q8_0_K;

ggml_tensor_repacking_mode_t ggml_tensor_repacking_scheme(ggml_tensor_repacking_mode_t t);

void ggml_set_tensor_repacking_mode(ggml_tensor_repacking_mode_t type);

enum ggml_type ggml_repack_tensor(struct ggml_tensor *tensor);


#ifdef __cplusplus
}
#endif

