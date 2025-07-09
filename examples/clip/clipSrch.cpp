#pragma warning (disable:4100) // unref formal param
#pragma warning (disable:4127) // cond expr is constant
#pragma warning (disable:4242) // conversion from 'int' to 'unsigned short'
#pragma warning (disable:4244) // conversion from 'uint32_t' to 'uint16_t'
#pragma warning (disable:4245) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4267) // conversion from 'int64_t' to 'int'
#pragma warning (disable:4505) // unref function with internal linkage removed

#include "clip.h"
#include "common-clip.h"

#include "hnswlib/hnswlib.h"

#include <fstream>
#include <filesystem>

#if _WIN32

#include <string.h>
#include <windows.h>

static int64_t timer_freq = 0, timer_start = 0;
void timer_init(void) {
    if (!timer_freq) {
        LARGE_INTEGER t;
        QueryPerformanceFrequency(&t);
        timer_freq = t.QuadPart;

        // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
        // and the uptime is high enough.
        // We subtract the program start time to reduce the likelihood of that happening.
        QueryPerformanceCounter(&t);
        timer_start = t.QuadPart;
    }
}

int64_t timer_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart - timer_start) * 1000000) / timer_freq;
}

#else

#include <strings.h>

#endif // _WIN32

struct my_app_params {
    int32_t n_threads{1};
    std::string model{"./clip-vit-base-patch32_ggml-model-f16.gguf"};
    std::string vecdb{"./hnswlib_images.bin"};
    std::string filepaths{"./hnswlib_images.paths"};
    int32_t verbose{0};
    // TODO: index dir

    std::string search_text;
    std::string img_path;
    std::string search_dir;

    int32_t n_results{5};
};

void my_print_help(int argc, char ** argv, my_app_params & params) {
    printf("Usage: %s [options] <search string or /path/to/query/image>\n", argv[0]);
    printf("\nOptions:\n");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: overwrite path to model. Read from images.paths by default.\n");
    printf("  -d <path>, --db <path>: overwrite path to vecdb\n");
    printf("  -f <path>, --filepaths <path>: overwrite path to image file paths\n");
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
    printf("  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: %d\n",
           params.verbose);
    printf("  -s <path>, --searchdir <path>: directory of files to be searched in the database\n");
    printf("  -n N, --results N: Number of results to display. Default: %d\n", params.n_results);
}

// returns success
bool my_app_params_parse(int argc, char ** argv, my_app_params & params) {
    bool invalid_param = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-m" || arg == "--model") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.model = argv[i];
        } else if (arg == "-d" || arg == "--db") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.vecdb = argv[i];
        } else if (arg == "-f" || arg == "--filepaths") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.filepaths = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-n" || arg == "--results") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_results = std::stoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.verbose = std::stoi(argv[i]);
        } else if (arg == "-h" || arg == "--help") {
            my_print_help(argc, argv, params);
            exit(0);
        } else if (arg == "-s" || arg == "--search_dir") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.search_dir = argv[i];
        } else if (arg == "-repack-ggml") {
            ggml_set_tensor_repacking_mode(TENSOR_REPACKING_MODE_GGML);
        } else if (arg == "-repack-xbox") {
            ggml_set_tensor_repacking_mode(TENSOR_REPACKING_MODE_XBOX);
        } else if (arg.find('-') == 0) {
            if (i != 0) {
                printf("%s: unrecognized argument: %s\n", __func__, arg.c_str());
                return false;
            }
        } else {
            // assume search string from here on out
            if (i == argc - 1 && is_image_file_extension(arg)) {
                params.img_path = arg;
            } else {
                params.search_text = arg;
                for (++i; i < argc; i++) {
                    params.search_text += " ";
                    params.search_text += argv[i];
                }
            }
        }
    }

    return !(invalid_param || (params.search_text.empty() && params.img_path.empty() && params.search_dir.empty()));
}

int main(int argc, char ** argv) {
    ggml_time_init();

    my_app_params params;
    if (!my_app_params_parse(argc, argv, params)) {
        my_print_help(argc, argv, params);
        return 1;
    }

    // load model path
    std::ifstream image_file_index_file(params.filepaths.c_str(), std::ios::binary);
    std::string cached_model_path;
    std::getline(image_file_index_file, cached_model_path);
    if (params.model.empty()) {
        params.model = cached_model_path;
        if (params.verbose >= 1) {
            printf("[%s]: model arg is empty - use cached model file: '%s'\n", __func__, params.model.c_str());
        }
    } 
    else {
        std::filesystem::path cached_model_fullpath = std::filesystem::absolute(cached_model_path);
        std::filesystem::path model_fullpath = std::filesystem::absolute(params.model);
        std::string cached_model_name = cached_model_fullpath.filename().string();
        std::string model_name = model_fullpath.filename().string();
#if defined(_WIN32)
        if (_strnicmp(model_name.c_str(), cached_model_name.c_str(), cached_model_name.length()) != 0) {
#else
        if (strncasecmp(model_name.c_str(), cached_model_name.c_str(), cached_model_name.length()) != 0) {
#endif // _WIN32
            printf("[%s]: *******************************************\n"
                   "[%s]: using alternative model from cmdline '%s'. \n"
                   "[%s]: The index database was created with model '%s'.\n"
                   "[%s]: Expect errors if the models are different. \n"
                   "[%s]: *******************************************\n",
                __func__, __func__, model_name.c_str(), __func__, 
                cached_model_name.c_str(), __func__, __func__);
        }
    }

    // load model
    auto clip_ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (!clip_ctx) {
        printf("%s: Unable to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    // load paths and embeddings database
    std::vector<std::string> image_file_index;

    // load image paths
    do {
        std::string line;
        std::getline(image_file_index_file, line);
        if (line.empty()) {
            break;
        }
        image_file_index.push_back(line);
    } while (image_file_index_file.good());

    const int vec_dim = clip_get_vision_hparams(clip_ctx)->projection_dim;
    std::vector<float> vec(vec_dim);

    std::string hnsw_path = params.vecdb.c_str();
    hnswlib::L2Space space(vec_dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    size_t max_elements = alg_hnsw->getMaxElements();
    size_t cur_elementCount = alg_hnsw->getCurrentElementCount();

    if (params.verbose >= 1) {
        printf("[%s]: load vector DB '%s' - max_elms(%zd)/cur_elms(%zd)\n", __func__, hnsw_path.c_str(),
            max_elements, cur_elementCount);
    }

    if (image_file_index.size() != cur_elementCount) {
        printf("%s: index files size mismatched - expected (%zd) - actual (%zd)\n", 
            __func__, image_file_index.size(), cur_elementCount);
    }

    std::vector<std::pair<float, hnswlib::labeltype>> results;

    int64_t t_start = ggml_time_us();

    if (!params.img_path.empty()) {
        printf("[%s]: searching DB for image '%s'\n", __func__, params.img_path.c_str());
        clip_image_u8 img0;
        if (!clip_image_load_from_file(params.img_path.c_str(), &img0)) {
            fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.img_path.c_str());
            clip_free(clip_ctx);
            return 1;
        }

        clip_image_f32 img_res;
        printf("[%s]: preprocessing image...\n", __func__);
        clip_image_preprocess(clip_ctx, &img0, &img_res);

        printf("[%s]: encoding image...\n", __func__);
        if (!clip_image_encode(clip_ctx, params.n_threads, &img_res, vec, true)) {
            fprintf(stderr, "%s: failed to encode image from '%s'\n", __func__, params.img_path.c_str());
            clip_free(clip_ctx);
            return 1;
        }
    
        printf("[%s]: KNN search image...\n", __func__);
        results = alg_hnsw->searchKnnCloserFirst(vec.data(), params.n_results);

        for (auto item: results) {
            printf("[%s]: distance: %f - [%s]\n", __func__, item.first, image_file_index.at(item.second).c_str());
        }
    
    } else if (!params.search_dir.empty()) {
        auto imgs_dir = get_dir_keyed_files(params.search_dir, 0);

        std::vector<float> encode_timing_ms;
        int error_count = 0;
        int match_count = 0;
        for (auto & entry : imgs_dir) {
            printf("[%s]: processing %zu files in '%s'\n", __func__, entry.second.size(), entry.first.c_str());
            size_t n_imgs = entry.second.size();
            for (int img_index = 0; img_index < n_imgs; img_index++) {
                clip_image_u8 img0;
                const std::string & img_path = entry.second[img_index];
                if (params.verbose >= 2) {
                    printf("    [%s]: processing image file '%s'\n", __func__, img_path.c_str());
                }

                if (!clip_image_load_from_file(img_path.c_str(), &img0)) {
                    fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
                    continue;
                }

                clip_image_f32 img_res;
                clip_image_preprocess(clip_ctx, &img0, &img_res);
        
                int64_t t0 = ggml_time_us();
                if (!clip_image_encode(clip_ctx, params.n_threads, &img_res, vec, true)) {
                    fprintf(stderr, "%s: failed to encode image from '%s'\n", __func__, img_path.c_str());
                    clip_free(clip_ctx);
                    return 1;
                }
                int64_t t1 = ggml_time_us();
                encode_timing_ms.push_back(((t1 - t0) / 1000.0));

                results = alg_hnsw->searchKnnCloserFirst(vec.data(), params.n_results);
                auto item = results[0];
                if (item.first != 0.0f) {
                    error_count++;
                    if (params.verbose >=1) {
                        printf("[%s]: '%s' not found in DB\n", __func__, img_path.c_str());
                    }
                } else {
                    match_count++;
                    if (params.verbose >=1) {
                        printf("[%s]: Located matching entry [%zd] - '%s' in DB\n", 
                            __func__, item.second, image_file_index.at(item.second).c_str());
                    }
                }
            }
        }

        // print encoding stats
        if (!encode_timing_ms.empty()) {
            float sum = 0;
            for (float val : encode_timing_ms) {
                sum += val;
            }
            float mean = sum / encode_timing_ms.size();

            float variance = 0;
            for (float_t val : encode_timing_ms) {
                variance += (val - mean) * (val - mean);
            } 
            variance /= encode_timing_ms.size(); // Use (data.size() - 1) for sample std dev
            float stddev = std::sqrt(variance);
            printf("[%s]: Average encoding time / stddev: %.2fms +/- %.2f\n", __func__, mean, stddev);
        }

        printf("[%s]: Matched successfully '%d' entries with '%d' error(s)\n", 
            __func__, match_count, error_count);

    } else {

        printf("[%s]: searching DB for string '%s'\n", __func__, params.search_text.c_str());
        clip_tokens tokens;
        clip_tokenize(clip_ctx, params.search_text.c_str(), &tokens);
        clip_text_encode(clip_ctx, params.n_threads, &tokens, vec, true);

        printf("[%s]: KNN search image...\n", __func__);
        results = alg_hnsw->searchKnnCloserFirst(vec.data(), params.n_results);
    
        if (params.verbose > 0) {
            printf("[%s]: search results - distance path:\n", __func__);
        }
        
        for (auto item: results) {
            printf("    distance: %f - [%zd]-<%s>\n", item.first, item.second, image_file_index.at(item.second).c_str());
        }
    }

    int64_t t_elapsed = ggml_time_us() - t_start;

    printf("\n[%s]: Elapsed time: %.2fs\n", __func__, t_elapsed / 1000.0 / 1000.0);
    print_tensor_op_perf_data(t_elapsed);

    clip_free(clip_ctx);
    delete alg_hnsw;

    return 0;
}
