#include "ggml.h"

#include <cstdio>
#include <cinttypes>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

template <typename T>
static std::string to_string(const T & val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

static bool gguf_ex_write(const std::string & fname) {
    struct gguf_context * ctx = gguf_init_empty();

    gguf_set_val_u8  (ctx, "some.parameter.uint8",    0x12);
    gguf_set_val_i8  (ctx, "some.parameter.int8",    -0x13);
    gguf_set_val_u16 (ctx, "some.parameter.uint16",   0x1234);
    gguf_set_val_i16 (ctx, "some.parameter.int16",   -0x1235);
    gguf_set_val_u32 (ctx, "some.parameter.uint32",   0x12345678);
    gguf_set_val_i32 (ctx, "some.parameter.int32",   -0x12345679);
    gguf_set_val_f32 (ctx, "some.parameter.float32",  0.123456789f);
    gguf_set_val_u64 (ctx, "some.parameter.uint64",   0x123456789abcdef0ull);
    gguf_set_val_i64 (ctx, "some.parameter.int64",   -0x123456789abcdef1ll);
    gguf_set_val_f64 (ctx, "some.parameter.float64",  0.1234567890123456789);
    gguf_set_val_bool(ctx, "some.parameter.bool",     true);
    gguf_set_val_str (ctx, "some.parameter.string",   "hello world");

    gguf_set_arr_data(ctx, "some.parameter.arr.i16", GGUF_TYPE_INT16,   std::vector<int16_t>{ 1, 2, 3, 4, }.data(), 4);
    gguf_set_arr_data(ctx, "some.parameter.arr.f32", GGUF_TYPE_FLOAT32, std::vector<float>{ 3.145f, 2.718f, 1.414f, }.data(), 3);
    gguf_set_arr_str (ctx, "some.parameter.arr.str",                    std::vector<const char *>{ "hello", "world", "!" }.data(), 3);

    struct ggml_init_params params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx_data = ggml_init(params);

    const int n_tensors = 10;

    // tensor infos
    for (int i = 0; i < n_tensors; ++i) {
        const std::string name = "tensor_" + to_string(i);

        int64_t ne[GGML_MAX_DIMS] = { 1 };
        int32_t n_dims = rand() % GGML_MAX_DIMS + 1;

        for (int j = 0; j < n_dims; ++j) {
            ne[j] = rand() % 10 + 1;
        }

        struct ggml_tensor * cur = ggml_new_tensor(ctx_data, GGML_TYPE_F32, n_dims, ne);
        ggml_set_name(cur, name.c_str());

        {
            float * data = (float *) cur->data;
            for (int j = 0; j < ggml_nelements(cur); ++j) {
                data[j] = 100 + i;
            }
        }

        gguf_add_tensor(ctx, cur);
    }

    gguf_write_to_file(ctx, fname.c_str(), false);

    printf("%s: wrote file '%s;\n", __func__, fname.c_str());

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

std::string ggml_type_str_from_ggml_type(const ggml_type ggml_t) {
    std::string type_str;

    switch (ggml_t) {
        case GGML_TYPE_F32: 
            type_str = "GGML_TYPE_F32";
            break;
        case GGML_TYPE_F16:
            type_str = "GGML_TYPE_F16";
            break;
        case GGML_TYPE_Q4_0:
            type_str = "GGML_TYPE_Q4_0";
            break;
        case GGML_TYPE_Q4_1:
            type_str = "GGML_TYPE_Q4_1";
            break;
        case GGML_TYPE_Q5_0:
            type_str = "GGML_TYPE_Q5_0";
            break;
        case GGML_TYPE_Q5_1:
            type_str = "GGML_TYPE_Q5_1";
            break;
        case GGML_TYPE_Q8_0:
            type_str = "GGML_TYPE_Q8_0";
            break;
        case GGML_TYPE_Q8_1:
            type_str = "GGML_TYPE_Q8_1";
            break;
        case GGML_TYPE_Q2_K:
            type_str = "GGML_TYPE_Q2_K";
            break;
        case GGML_TYPE_Q3_K:
            type_str = "GGML_TYPE_Q3_K";
            break;
        case GGML_TYPE_Q4_K:
            type_str = "GGML_TYPE_Q4_K";
            break;
        case GGML_TYPE_Q5_K:
            type_str = "GGML_TYPE_Q5_K";
            break;
        case GGML_TYPE_Q6_K:
            type_str = "GGML_TYPE_Q6_K";
            break;
        case GGML_TYPE_Q8_K:
            type_str = "GGML_TYPE_Q8_K";
            break;
        case GGML_TYPE_IQ2_XXS:
            type_str = "GGML_TYPE_IQ2_XXS";
            break;
        case GGML_TYPE_IQ2_XS:
            type_str = "GGML_TYPE_IQ2_XS";
            break;
        case GGML_TYPE_IQ3_XXS:
            type_str = "GGML_TYPE_IQ3_XXS";
            break;
        case GGML_TYPE_IQ1_S:
            type_str = "GGML_TYPE_IQ1_S";
            break;
        case GGML_TYPE_IQ4_NL:
            type_str = "GGML_TYPE_IQ4_NL";
            break;
        case GGML_TYPE_IQ3_S:
            type_str = "GGML_TYPE_IQ3_S";
            break;
        case GGML_TYPE_IQ2_S:
            type_str = "GGML_TYPE_IQ2_S";
            break;
        case GGML_TYPE_IQ4_XS:
            type_str = "GGML_TYPE_IQ4_XS";
            break;
        case GGML_TYPE_I8:
            type_str = "GGML_TYPE_I8";
            break;
        case GGML_TYPE_I16:
            type_str = "GGML_TYPE_I16";
            break;
        case GGML_TYPE_I32:
            type_str = "GGML_TYPE_I32";
            break;
        case GGML_TYPE_I64:
            type_str = "GGML_TYPE_I64";
            break;
        case GGML_TYPE_F64:
            type_str = "GGML_TYPE_F64";
            break;
        case GGML_TYPE_IQ1_M:
            type_str = "GGML_TYPE_IQ1_M";
            break;
        case GGML_TYPE_BF16:
            type_str = "GGML_TYPE_BF16";
            break;
        case GGML_TYPE_Q4_0_4_4:
            type_str = "GGML_TYPE_Q4_0_4_4";
            break;
        case GGML_TYPE_Q4_0_4_8:
            type_str = "GGML_TYPE_Q4_0_4_8";
            break;
        case GGML_TYPE_Q4_0_8_8:
            type_str = "GGML_TYPE_Q4_0_8_8";
            break;
        case GGML_TYPE_IQ1_BN:
            type_str = "GGML_TYPE_IQ1_BN";
            break;
        case GGML_TYPE_IQ2_BN:
            type_str = "GGML_TYPE_IQ2_BN";
            break;
        case GGML_TYPE_Q8_K64:
            type_str = "GGML_TYPE_Q8_K64";
            break;
        case GGML_TYPE_IQ2_K:
            type_str = "GGML_TYPE_IQ2_K";
            break;
        case GGML_TYPE_IQ3_K:
            type_str = "GGML_TYPE_IQ3_K";
            break;
        case GGML_TYPE_IQ4_K:
            type_str = "GGML_TYPE_IQ4_K";
            break;
        case GGML_TYPE_IQ5_K:
            type_str = "GGML_TYPE_IQ5_K";
            break;
        case GGML_TYPE_IQ6_K:
            type_str = "GGML_TYPE_IQ6_K";
            break;
        case GGML_TYPE_IQ2_TN:
            type_str = "GGML_TYPE_IQ2_TN";
            break;
        default: 
            type_str = "UNKNOWN";
            break;
    }

    return type_str;
}

// just read tensor info
static bool gguf_ex_read_0(const std::string & fname) {
    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    printf("%s: version:      %d\n", __func__, gguf_get_version(ctx));
    printf("%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    printf("%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        printf("%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            printf("%s: kv[%3d]: key = %s\n", __func__, i, key);
        }
    }

    // find kv string
    {
        const char * findkey = "some.parameter.string";

        const int keyidx = gguf_find_key(ctx, findkey);
        if (keyidx == -1) {
            printf("%s: find key: %s not found.\n", __func__, findkey);
        } else {
            const char * key_value = gguf_get_val_str(ctx, keyidx);
            printf("%s: find key: %s found, kv[%3d] value = %s\n", __func__, findkey, keyidx, key_value);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        printf("%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const ggml_type ggml_t = gguf_get_tensor_type(ctx, i);

            printf("%s: tensor[%3d]: name = %30s, type = %-16s\n", 
                __func__, i, name, ggml_type_str_from_ggml_type(ggml_t).c_str());
        }
    }

    gguf_free(ctx);

    return true;
}

// read and create ggml_context containing the tensors and their data
static bool gguf_ex_read_1(const std::string & fname, bool check_data) {
    struct ggml_context * ctx_data = NULL;

    struct gguf_init_params params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ &ctx_data,
    };

    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);

    printf("%s: version:      %d\n", __func__, gguf_get_version(ctx));
    printf("%s: alignment:   %zu\n", __func__, gguf_get_alignment(ctx));
    printf("%s: data offset: %zu\n", __func__, gguf_get_data_offset(ctx));

    // kv
    {
        const int n_kv = gguf_get_n_kv(ctx);

        printf("%s: n_kv: %d\n", __func__, n_kv);

        for (int i = 0; i < n_kv; ++i) {
            const char * key = gguf_get_key(ctx, i);

            printf("%s: kv[%3d]: key = %s\n", __func__, i, key);
        }
    }

    // tensor info
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        printf("%s: n_tensors: %d\n", __func__, n_tensors);

        for (int i = 0; i < n_tensors; ++i) {
            const char * name   = gguf_get_tensor_name  (ctx, i);
            const size_t offset = gguf_get_tensor_offset(ctx, i);
            const ggml_type ggml_t = gguf_get_tensor_type(ctx, i);

            printf("%s: tensor[%3d]: name = %30s, type = %-16s, offset = %12zu\n", 
                __func__, i, name, ggml_type_str_from_ggml_type(ggml_t).c_str(), offset);
        }
    }

    // data
    {
        const int n_tensors = gguf_get_n_tensors(ctx);

        for (int i = 0; i < n_tensors; ++i) {
            // printf("%s: reading tensor %d data\n", __func__, i);

            const char * name = gguf_get_tensor_name(ctx, i);

            struct ggml_tensor * cur = ggml_get_tensor(ctx_data, name);

            printf("%s: tensor[%3d]: n_dims = %d, name = %30s, data = %p\n", __func__, i, ggml_n_dims(cur), cur->name, cur->data);

            // print first 10 elements
            const float * data = (const float *) cur->data;

#if 0
            printf("%s data[:10] : ", name);
            for (int j = 0; j < MIN(10, ggml_nelements(cur)); ++j) {
                printf("%f ", data[j]);
            }
            printf("\n\n");

            // check data
            if (check_data) {
                const float * data = (const float *) cur->data;
                for (int j = 0; j < ggml_nelements(cur); ++j) {
                    if (data[j] != 100 + i) {
                        fprintf(stderr, "%s: tensor[%d]: data[%d] = %f\n", __func__, i, j, data[j]);
                        gguf_free(ctx);
                        return false;
                    }
                }
            }
#endif
        }
    }

    printf("%s: ctx_data size: %zu\n", __func__, ggml_get_mem_size(ctx_data));

    ggml_free(ctx_data);
    gguf_free(ctx);

    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s data.gguf r|w [c]\n", argv[0]);
        printf("r: read data.gguf file\n");
        printf("w: write data.gguf file\n");
        printf("c: check tensor data\n");
        return -1;
    }
    bool check_data = false;
    if (argc == 4) {
        check_data = true;
    }

    const std::string fname(argv[1]);

    if (argc == 2) {
        if (!gguf_ex_read_0(fname)) {
            printf("failed to check gguf file\n");
        };
        return 0;
    }
    
    const std::string mode (argv[2]);
    bool default_mode = false;

    if ((mode != "r") && (mode != "w")) {
        default_mode = true;
        printf("no mode specified - default to 'r'\n");
    }

    if (mode == "w") {
        if (!gguf_ex_write(fname)) {
            printf("failed to write gguf file\n");
        }
    } else if (default_mode || (mode == "r")) {
#if 0
        if (!gguf_ex_read_0(fname)) {
            printf("failed to read gguf file\n");
        }
#endif
        if (!gguf_ex_read_1(fname, check_data)) {
            printf("failed to check gguf file\n");
        };
    }

    return 0;
}
