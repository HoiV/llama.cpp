#include "clip.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#include <string>
#include <fstream>
#include <regex>
#include <vector>

// *ONLY* For CLIP support (from an older GGML commit)

static size_t clip_quantize_q4_0(const float * src, void * dst, int n, int k, int64_t * hist) {
    GGML_ASSERT(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int b = 0; b < n; b += k) {
        block_q4_0 * y = (block_q4_0 *) dst + b/QK4_0;

        quantize_row_q4_0(src + b, y, k);

        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < QK4_0; j += 2) {
                const uint8_t vi0 = y[i].qs[j/2] & 0x0F;
                const uint8_t vi1 = y[i].qs[j/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_0*sizeof(block_q4_0));
}

size_t clip_quantize_q4_1(const float * src, void * dst, int n, int k, int64_t * hist) {
    GGML_ASSERT(k % QK4_1 == 0);
    const int nb = k / QK4_1;

    for (int b = 0; b < n; b += k) {
        block_q4_1 * y = (block_q4_1 *) dst + b/QK4_1;

        quantize_row_q4_1(src + b, y, k);

        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < QK4_1; j += 2) {
                const uint8_t vi0 = y[i].qs[j/2] & 0x0F;
                const uint8_t vi1 = y[i].qs[j/2] >> 4;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK4_1*sizeof(block_q4_1));
}

size_t clip_quantize_q4_K(const float * src, void * dst, int n, int k, int64_t * hist) {
    GGML_UNUSED(hist);

    GGML_ASSERT(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int b = 0; b < n; b += k) {
        block_q4_K * y = (block_q4_K *) dst + b/QK_K;

        quantize_row_q4_K(src + b, y, k);
    }

    return (n/QK_K*sizeof(block_q4_K));
}

size_t clip_quantize_q5_0(const float * src, void * dst, int n, int k, int64_t * hist) {
    GGML_ASSERT(k % QK5_0 == 0);
    const int nb = k / QK5_0;

    for (int b = 0; b < n; b += k) {
        block_q5_0 * y = (block_q5_0 *)dst + b/QK5_0;

        quantize_row_q5_0_reference(src + b, y, k);

        for (int i = 0; i < nb; i++) {
            uint32_t qh;
            memcpy(&qh, &y[i].qh, sizeof(qh));

            for (int j = 0; j < QK5_0; j += 2) {
                const uint8_t vh0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
                const uint8_t vh1 = ((qh & (1u << (j + 16))) >> (j + 12));

                // cast to 16 bins
                const uint8_t vi0 = ((y[i].qs[j/2] & 0x0F) | vh0) / 2;
                const uint8_t vi1 = ((y[i].qs[j/2] >>   4) | vh1) / 2;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK5_0*sizeof(block_q5_0));
}

size_t clip_quantize_q5_1(const float * src, void * dst, int n, int k, int64_t * hist) {
    GGML_ASSERT(k % QK5_1 == 0);
    const int nb = k / QK5_1;

    for (int b = 0; b < n; b += k) {
        block_q5_1 * y = (block_q5_1 *)dst + b/QK5_1;

        quantize_row_q5_1_reference(src + b, y, k);

        for (int i = 0; i < nb; i++) {
            uint32_t qh;
            memcpy(&qh, &y[i].qh, sizeof(qh));

            for (int j = 0; j < QK5_1; j += 2) {
                const uint8_t vh0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
                const uint8_t vh1 = ((qh & (1u << (j + 16))) >> (j + 12));

                // cast to 16 bins
                const uint8_t vi0 = ((y[i].qs[j/2] & 0x0F) | vh0) / 2;
                const uint8_t vi1 = ((y[i].qs[j/2] >>   4) | vh1) / 2;

                hist[vi0]++;
                hist[vi1]++;
            }
        }
    }

    return (n/QK5_1*sizeof(block_q5_1));
}

size_t clip_quantize_q8_0(const float * src, void * dst, int n, int k, int64_t * hist) {
    GGML_ASSERT(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int b = 0; b < n; b += k) {
        block_q8_0 * y = (block_q8_0 *)dst + b/QK8_0;

        quantize_row_q8_0(src + b, y, k);

        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < QK8_0; ++j) {
                const int8_t vi = y[i].qs[j];

                hist[vi/16 + 8]++;
            }
        }
    }

    return (n/QK8_0*sizeof(block_q8_0));
}

bool clip_model_quantize(const char * fname_inp, const char * fname_out, const int itype) {

    ggml_type type = GGML_TYPE_Q4_1;

    switch (itype) {
    case 2:
        type = GGML_TYPE_Q4_0;
        break;
    case 3:
        type = GGML_TYPE_Q4_1;
        break;
    case 4:
        type = GGML_TYPE_Q4_K;
        break;
    case 6:
        type = GGML_TYPE_Q5_0;
        break;
    case 7:
        type = GGML_TYPE_Q5_1;
        break;
    case 8:
        type = GGML_TYPE_Q8_0;
        break;
    case 9: 
        type = GGML_TYPE_BF16;
        break;
    default:
        fprintf(stderr, "%s: invalid quantization type %d\n", __func__, itype);
        return false;
    };

    auto ctx_clip = clip_model_load(fname_inp, 2);
    const auto & ctx_gguf_src = ctx_clip->ctx_gguf;
    const auto & ctx_model = ctx_clip->ctx_model;

    auto ctx_gguf_out = gguf_init_empty();
    gguf_set_kv(ctx_gguf_out, ctx_gguf_src);
    gguf_set_val_u32(ctx_gguf_out, "general.quantization_version", GGML_QNT_VERSION);
    gguf_set_val_u32(ctx_gguf_out, "general.file_type", itype);

    auto fout = std::ofstream(fname_out, std::ios::binary);

    const int n_tensors = gguf_get_n_tensors(ctx_gguf_src);

    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx_gguf_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_model, name);
        gguf_add_tensor(ctx_gguf_out, cur);
    }

    const size_t meta_size = gguf_get_meta_size(ctx_gguf_out);
    for (size_t i = 0; i < meta_size; ++i) {
        fout.put(0);
    }

    // regexes of tensor names to be quantized
    const std::vector<std::string> k_names = {
        ".*weight",
    };

    std::vector<uint8_t> read_data(512);
    std::vector<uint8_t> work(512);
    std::vector<float> conv_buf(512);
    std::vector<int64_t> hist_all(1 << 4, 0);
    size_t total_size_org = 0;
    size_t total_size_new = 0;

    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = gguf_get_tensor_name(ctx_gguf_src, i);
        struct ggml_tensor * cur = ggml_get_tensor(ctx_model, name.c_str());

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;

        bool quantize = false;
        for (const auto & s : k_names) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (ggml_n_dims(cur) == 2);

        if (quantize) {
            new_type = type;
            const size_t n_elms = ggml_nelements(cur);
            float * f32_data;

            switch (cur->type) {
            case GGML_TYPE_F32:
                f32_data = (float *)cur->data;
                break;
            case GGML_TYPE_F16:
                if (conv_buf.size() < n_elms) {
                    conv_buf.resize(n_elms);
                }
                for (int j = 0; j < n_elms; ++j) {
                    conv_buf[j] = ggml_fp16_to_fp32(((ggml_fp16_t *)cur->data)[j]);
                }
                f32_data = (float *)conv_buf.data();
                break;
            default:
                printf("Please use an input file in f32 or f16\n");
                return false;
            }

            if (work.size() < n_elms * 4) {
                work.resize(n_elms * 4);
            }
            new_data = work.data();

            std::vector<int64_t> hist_cur(1 << 4, 0);

            switch (new_type) {
            case GGML_TYPE_Q4_0: {
                new_size = clip_quantize_q4_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
            case GGML_TYPE_Q4_1: {
                new_size = clip_quantize_q4_1(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
                case GGML_TYPE_Q4_K: {
                    new_size = clip_quantize_q4_K(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                    } break;
                case GGML_TYPE_Q5_0: {
                new_size = clip_quantize_q5_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
            case GGML_TYPE_Q5_1: {
                new_size = clip_quantize_q5_1(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
            case GGML_TYPE_Q8_0: {
                new_size = clip_quantize_q8_0(f32_data, new_data, n_elms, cur->ne[0], hist_cur.data());
                } break;
            case GGML_TYPE_BF16: {
                ggml_bf16_t * bf16_data = (ggml_bf16_t *) new_data;
                for (int j = 0; j < n_elms; ++j) {
                    bf16_data[j] = ggml_fp32_to_bf16(((float *)f32_data)[j]);
                }
                new_size = n_elms * sizeof(ggml_bf16_t);
                } break;
                default: {
                fprintf(stderr, "%s: unsupported quantization type %d\n", __func__, new_type);
                return false;
                }
            }

            for (int j = 0; j < hist_cur.size(); ++j) {
                hist_all[j] += hist_cur[j];
            }
        } else {
            new_type = cur->type;
            new_data = cur->data;
            new_size = ggml_nbytes(cur);
        }
        const size_t orig_size = ggml_nbytes(cur);
        total_size_org += orig_size;
        total_size_new += new_size;
        gguf_set_tensor_type(ctx_gguf_out, name.c_str(), new_type);
        gguf_set_tensor_data(ctx_gguf_out, name.c_str(), new_data, new_size);
        fout.write((const char *)new_data, new_size);
        size_t pad = GGML_PAD(new_size, gguf_get_alignment(ctx_gguf_out)) - new_size;
        for (int j = 0; j < pad; ++j) {
            fout.put(0);
        }

        printf("%s: n_dims = %d | quantize=%d | size = %f MB -> %f MB\n", name.c_str(), ggml_n_dims(cur), quantize,
               orig_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
    }

    // go back to beginning of file and write the updated metadata
    fout.seekp(0, std::ios::beg);
    std::vector<uint8_t> meta(meta_size);
    gguf_get_meta_data(ctx_gguf_out, meta.data());
    fout.write((const char *)meta.data(), meta_size);

    fout.close();

    clip_free(ctx_clip);
    gguf_free(ctx_gguf_out);

    {
        printf("%s: original size   = %8.2f MB\n", __func__, total_size_org / 1024.0 / 1024.0);
        printf("%s: quantized size  = %8.2f MB\n", __func__, total_size_new / 1024.0 / 1024.0);

        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); ++i) {
            sum_all += hist_all[i];
        }

        if (sum_all != 0) {
            printf("%s: hist: ", __func__);
            for (size_t i = 0; i < hist_all.size(); ++i) {
                printf("%5.3f ", hist_all[i] / (float)sum_all);
            }
            printf("\n");
        }
    }

    return true;
}

void print_usage(int argc, char ** argv) {

    fprintf(stderr, "usage: %s /path/to/ggml-model-f32.gguf /path/to/ggml-model-quantized.gguf type\n", argv[0]);
    fprintf(stderr, "  type = 2 - q4_0\n");
    fprintf(stderr, "  type = 3 - q4_1\n");
    fprintf(stderr, "  type = 4 - q4_K\n");
    fprintf(stderr, "  type = 6 - q5_0\n");
    fprintf(stderr, "  type = 7 - q5_1\n");
    fprintf(stderr, "  type = 8 - q8_0\n");
    fprintf(stderr, "  type = 9 - bf16\n");
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        print_usage(argc, argv);
        return 1;
    }

    const std::string fname_inp = argv[1];
    const std::string fname_out = argv[2];

    const int itype = atoi(argv[3]);
    if (itype != 2 && itype != 3 && itype != 4 && itype != 6 && itype != 7 && itype != 8 && itype != 9) {
        print_usage(argc, argv);
        return 1;
    }

    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_quantize_us = 0;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!clip_model_quantize(fname_inp.c_str(), fname_out.c_str(), itype)) {
            fprintf(stderr, "%s: failed to quantize model from '%s'\n", __func__, fname_inp.c_str());
            return 1;
        }

        t_quantize_us = ggml_time_us() - t_start_us;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n");
        printf("%s: quantize time = %8.2f ms\n", __func__, t_quantize_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    return 0;
}