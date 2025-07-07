#ifndef CLIP_H
#define CLIP_H

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-repack.h"

#include <vector>

#include <stdint.h>
#include <stddef.h>
#include <vector>
#include <string>
#include <map>

#ifdef __cplusplus
extern "C" {
#endif

struct clip_text_hparams {
    int32_t n_vocab;
    int32_t num_positions;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
    float eps;
};

struct clip_vision_hparams {
    int32_t image_size;
    int32_t patch_size;
    int32_t hidden_size;
    int32_t n_intermediate;
    int32_t projection_dim;
    int32_t n_head;
    int32_t n_layer;
    float eps;
};

typedef int32_t clip_vocab_id;
struct clip_tokens {
    clip_vocab_id * data;
    size_t size;
};

//
// Vocab utils
//

struct clip_vocab {
    using id = clip_vocab_id;
    using token = std::string;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    //    void add_special_token(const std::string & token);
};

//
// clip layers
//

struct clip_layer {
    // attention
    struct ggml_tensor * k_w;
    struct ggml_tensor * k_b;
    struct ggml_tensor * q_w;
    struct ggml_tensor * q_b;
    struct ggml_tensor * v_w;
    struct ggml_tensor * v_b;

    struct ggml_tensor * o_w;
    struct ggml_tensor * o_b;

    // layernorm 1
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_1_b;

    // ff
    struct ggml_tensor * ff_i_w;
    struct ggml_tensor * ff_i_b;

    struct ggml_tensor * ff_o_w;
    struct ggml_tensor * ff_o_b;

    // layernorm 2
    struct ggml_tensor * ln_2_w;
    struct ggml_tensor * ln_2_b;
};

struct clip_text_model {
    struct clip_text_hparams hparams;

    // embeddings
    struct ggml_tensor * token_embeddings;
    struct ggml_tensor * position_embeddings;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;
};

struct clip_vision_model {
    struct clip_vision_hparams hparams;

    // embeddings
    struct ggml_tensor * class_embedding;
    struct ggml_tensor * patch_embeddings;
    struct ggml_tensor * position_embeddings;

    struct ggml_tensor * pre_ln_w;
    struct ggml_tensor * pre_ln_b;

    std::vector<clip_layer> layers;

    struct ggml_tensor * post_ln_w;
    struct ggml_tensor * post_ln_b;

    struct ggml_tensor * projection;
};

// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct clip_buffer {
    uint8_t * data = nullptr;
    size_t size = 0;

    void resize(size_t size) {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~clip_buffer() { delete[] data; }
};

struct clip_ctx {
    bool has_text_encoder = false;
    bool has_vision_encoder = false;
    struct clip_text_model text_model;
    struct clip_vision_model vision_model;
    struct clip_vocab vocab;
    float image_mean[3];
    float image_std[3];
    bool use_gelu = false;
    int32_t ftype = 1;
    struct ggml_context * ctx_model = nullptr;
    struct ggml_context * ctx_gf    = nullptr;
    struct gguf_context * ctx_gguf  = nullptr;

    // memory buffers to evaluate the model
    struct clip_buffer buf_compute;
    ggml_backend_buffer_t params_buffer = nullptr;
    ggml_backend_t backend              = nullptr;
    ggml_gallocr_t compute_alloc        = nullptr;
};

struct clip_ctx * clip_model_load(const char * fname, const int verbosity);

void clip_free(struct clip_ctx * ctx);

struct clip_text_hparams * clip_get_text_hparams(struct clip_ctx * ctx);
struct clip_vision_hparams * clip_get_vision_hparams(struct clip_ctx * ctx);

// RGB uint8 image
struct clip_image_u8 {
    int nx;
    int ny;
    uint8_t * data;
    size_t size;
};

// RGB float32 image (NHWC)
// Memory layout: RGBRGBRGB...
struct clip_image_f32 {
    int nx;
    int ny;
    float * data;
    size_t size;
};

struct clip_image_u8_batch {
    struct clip_image_u8 * data;
    size_t size;
};

struct clip_image_f32_batch {
    struct clip_image_f32 * data;
    size_t size;
};

bool clip_tokenize(const struct clip_ctx * ctx, const char * text, struct clip_tokens * tokens);

struct clip_image_u8 * clip_image_u8_make();
struct clip_image_f32 * clip_image_f32_make();

void clip_image_u8_clean(struct clip_image_u8 * img);
void clip_image_f32_clean(struct clip_image_f32 * res);

void clip_image_u8_free(struct clip_image_u8 * img);
void clip_image_f32_free(struct clip_image_f32 * res);

bool clip_image_load_from_file(const char * fname, struct clip_image_u8 * img);
bool clip_image_preprocess(const struct clip_ctx * ctx, 
    const struct clip_image_u8 * img, struct clip_image_f32 * res);

bool clip_text_encode(const struct clip_ctx * ctx, const int n_threads, const struct clip_tokens * tokens, 
    std::vector<float> & vec, const bool normalize);
bool clip_image_encode(const struct clip_ctx * ctx, const int n_threads, struct clip_image_f32 * img, 
    std::vector<float> & vec, const bool normalize);

void clip_image_batch_preprocess(const struct clip_ctx * ctx, const int n_threads,
    const struct clip_image_u8_batch * img_inputs, struct clip_image_f32_batch * imgs_resized);
bool clip_image_batch_encode(const struct clip_ctx * ctx, const int n_threads, 
    const struct clip_image_f32_batch * imgs, std::vector<float> & vec, bool normalize = false);

// bool image_normalize(const clip_image_u8 *img, clip_image_f32 *res);

bool clip_compare_text_and_image(const struct clip_ctx * ctx, const int n_threads, const char * text,
    const struct clip_image_u8 * image, float * score);
float clip_similarity_score(const float * vec1, const float * vec2, const int vec_dim);
bool softmax_with_sorting(float * arr, const int length, float * sorted_scores, int * indices);
bool clip_zero_shot_label_image(struct clip_ctx * ctx, const int n_threads, const struct clip_image_u8 * input_img,
    const char ** labels, const size_t n_labels, float * scores, int * indices);

bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype);

#ifdef __cplusplus
}
#endif

#endif // CLIP_H
