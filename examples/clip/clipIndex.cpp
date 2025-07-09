#pragma warning (disable:4100) // unref formal param
#pragma warning (disable:4127) // cond expr is constant
#pragma warning (disable:4242) // conversion from 'int' to 'unsigned short'
#pragma warning (disable:4244) // conversion from 'uint32_t' to 'uint16_t'
#pragma warning (disable:4245) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4267) // conversion from 'int64_t' to 'int'
#pragma warning (disable:4505) // unref function with internal linkage removed

#include <fstream>
#include <vector>

#include "clip.h"
#include "common-clip.h"

#include "hnswlib/hnswlib.h"

#if _WIN32

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

#endif // _WIN32

struct my_app_params {
    int32_t n_batch{1};
    int32_t n_threads{1};
    // std::string model{"./clip-vit-base-patch32_ggml-model-f16.gguf"};
    std::string model{"./clip-vit-L-336-f32.gguf"};
    std::string vecdb{"./hnswlib_images.bin"};
    std::string filepaths{"./hnswlib_images.paths"};
    int32_t verbose{0};
    std::vector<std::string> image_directories;
};
my_app_params params;

std::vector<std::string> image_file_index;

void my_print_help(int argc, char ** argv, my_app_params & params) {
    GGML_UNUSED(argc);
    printf("Usage: %s [options] dir/with/pictures [more/dirs]\n", argv[0]);
    printf("\nOptions:\n");
    printf("  -h, --help: Show this message and exit\n");
    printf("  -m <path>, --model <path>: path to model. Default: %s\n", params.model.c_str());
    printf("  -d <path>, --db <path>: overwrite path to vecdb\n");
    printf("  -f <path>, --filepaths <path>: overwrite path to image file paths\n");
    printf("  -b N, --batch N: Number of items to process together in batch mode. Default: 1\n");
    printf("  -t N, --threads N: Number of threads to use for inference. Default: %d\n", params.n_threads);
    printf("  -v <level>, --verbose <level>: Control the level of verbosity. 0 = minimum, 2 = maximum. Default: %d\n",
           params.verbose);
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
        } else if (arg == "-b" || arg == "--batch") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_batch = std::stoi(argv[i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.n_threads = std::stoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            if (++i >= argc) {
                invalid_param = true;
                break;
            }
            params.verbose = std::stoi(argv[i]);
        } else if (arg == "-repack-ggml") {
            ggml_set_tensor_repacking_mode(TENSOR_REPACKING_MODE_GGML);
        } else if (arg == "-repack-xbox") {
            ggml_set_tensor_repacking_mode(TENSOR_REPACKING_MODE_XBOX);
        } else if (arg == "-h" || arg == "--help") {
            my_print_help(argc, argv, params);
            exit(0);
        } else if (arg.find('-') == 0) {
            if (i != 0) {
                printf("%s: unrecognized argument: %s\n", __func__, arg.c_str());
                return false;
            }
        } else {
            // assume image directory
            params.image_directories.push_back(argv[i]);
        }
    }

    return !(invalid_param || params.image_directories.empty());
}

void testVecDb() {

    struct clip_ctx * clip_ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (clip_ctx == nullptr) {
        printf("%s: Unable  to load model from %s\n", __func__, params.model.c_str());
        return;
    }

    const size_t vec_dim = clip_get_vision_hparams(clip_ctx)->projection_dim;
    hnswlib::L2Space space2(vec_dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space2, params.vecdb);
    size_t max_elements = alg_hnsw->getMaxElements();
    size_t cur_elementCount = alg_hnsw->getCurrentElementCount();

    if (params.verbose >= 2) {
        printf("[%s]: load vector DB '%s' - max_elms(%zd)/cur_elms(%zd)\n", __func__, params.vecdb.c_str(),
            max_elements, cur_elementCount);
    }
    
    std::string test_img_path = image_file_index[0];
    printf("[%s]: searching DB for image '%s'\n", __func__, test_img_path.c_str());
    clip_image_u8 img0;
    if (!clip_image_load_from_file(test_img_path.c_str(), &img0)) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, test_img_path.c_str());
        return;
    }

    std::vector<float> testvec(vec_dim);
    clip_image_f32 img_res;
    int n_threads = params.n_threads;
    clip_image_preprocess(clip_ctx, &img0, &img_res);
    int64_t t0 = ggml_time_us();
    if (!clip_image_encode(clip_ctx, n_threads, &img_res, testvec, true)) {
        fprintf(stderr, "%s: failed to encode image from '%s'\n", __func__, test_img_path.c_str());
        return;
    }
    int64_t t1 = ggml_time_us();
    printf("[%s]: encoding time = %9.2fms\n", __func__, (t1 - t0) / 1000.0);

    // Query the elements for themselves and measure recall
    printf("[%s]: querying specific data for test image [%s]\n", __func__, test_img_path.c_str());
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(testvec.data(), 1);
    hnswlib::labeltype label = result.top().second;
    printf("[%s]: shortest distance=%f - resulting file=%s\n", __func__, result.top().first, 
        image_file_index.at(label).c_str());

    // Query the elements in batch
    printf("[%s]: querying closest data for test image [%s]\n", __func__, test_img_path.c_str());
    int k = 5;
    std::vector<std::pair<float, hnswlib::labeltype>> results = alg_hnsw->searchKnnCloserFirst(testvec.data(), k);
    for (auto item: results) {
        printf("[%s]: distance: %f - [%s]\n", __func__, item.first, image_file_index.at(item.second).c_str());
    }

    clip_free(clip_ctx);
    delete alg_hnsw;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    if (!my_app_params_parse(argc, argv, params)) {
        my_print_help(argc, argv, params);
        return 1;
    }

    const int64_t t_start = ggml_time_us();

    struct clip_ctx * clip_ctx = clip_model_load(params.model.c_str(), params.verbose);
    if (clip_ctx == NULL) {
        printf("%s: Unable  to load model from %s\n", __func__, params.model.c_str());
        return 1;
    }

    const size_t vec_dim = clip_get_vision_hparams(clip_ctx)->projection_dim;
    size_t batch_size = params.n_batch;
    if (params.verbose >= 1) {
        printf("%s: working with batch size %zd\n", __func__, batch_size);
    }

    int max_elements = 10000;   // Maximum number of elements, should be known beforehand

    // if dataset is big and memory is a concern then NHSW need to be chosen. 
    // As well as in NMSLIB, we chose the M parameter. The 4 <= M <= 64 is 
    // the number of links per vector, higher is more accurate but uses more RAM. 
    // The memory usage is (d * 4 + M * 2 * 4) bytes per vector.
    
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(vec_dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    size_t label = 0;

    std::vector<float> vec(vec_dim * batch_size);
    std::vector<clip_image_u8> img_inputs(batch_size);
    std::vector<clip_image_f32> imgs_resized(batch_size);

    std::vector<float> encode_timing_ms;

    // search for images in path and write embeddings to database
    for (const auto & base_dir : params.image_directories) {
        printf("[%s]: starting base dir scan of '%s'\n", __func__, base_dir.c_str());
        auto results = get_dir_keyed_files(base_dir, 0);

        for (auto & entry : results) {
            printf("[%s]: processing %zu files in '%s'\n", __func__, entry.second.size(), entry.first.c_str());

            size_t n_batches = (entry.second.size() / batch_size);

            vec.resize(vec_dim * batch_size);
            img_inputs.resize(batch_size);
            imgs_resized.resize(batch_size);

            if (params.verbose >= 1) {
                printf("[%s]: processing '%zd' batches of '%zd' each...\n", __func__, n_batches, batch_size);
            }
            size_t img_index = 0;
            for (size_t i = 0; i < n_batches; i++) {
                if (params.verbose >= 2) {
                    printf("[%s]: Batch %zd\n", __func__, i+1);
                }
                for (size_t ib = 0; ib < batch_size; ib++) {
                    img_index = ib + (i * batch_size);
                    const std::string & img_path = entry.second[img_index];
                    if (params.verbose >= 2) {
                        printf("    [%s]: processing image file '%s'\n", __func__, img_path.c_str());
                    }

                    if (!clip_image_load_from_file(img_path.c_str(), &img_inputs[ib % batch_size])) {
                        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
                        continue;
                    }

                    image_file_index.push_back(img_path);
                }

                auto img_inputs_batch = clip_image_u8_batch{};
                img_inputs_batch.data = img_inputs.data();
                img_inputs_batch.size = img_inputs.size();

                auto imgs_resized_batch = clip_image_f32_batch{};
                imgs_resized_batch.data = imgs_resized.data();
                imgs_resized_batch.size = imgs_resized.size();

                clip_image_batch_preprocess(clip_ctx, params.n_threads, &img_inputs_batch, &imgs_resized_batch);
                int64_t t0 = ggml_time_us();
                clip_image_batch_encode(clip_ctx, params.n_threads, &imgs_resized_batch, vec, true);
                int64_t t1 = ggml_time_us();
                if (params.verbose == 1) {
                    printf(".");
                }
                encode_timing_ms.push_back(((t1 - t0) / 1000.0) / batch_size);

                // add image vectors to the database
                for (size_t b = 0; b < batch_size; b++) {
                    alg_hnsw->addPoint(vec.data() + b * vec_dim, label++);
                    // embd_index.add(label++, {vec.data() + b * vec_dim, vec_dim});
                }
            }

            if (params.verbose == 1) {
                printf("\n");
            }

            // process leftover if needed
            const size_t leftover = entry.second.size() - (n_batches * batch_size);
            if (leftover > 0) {
                if (params.verbose >= 1) {
                    printf("[%s]: processing leftovers [%zd]...\n", __func__, leftover);
                }
                img_inputs.resize(leftover);
                imgs_resized.resize(leftover);

                for (size_t il = (n_batches * batch_size); il < entry.second.size(); il++) {
                    const std::string & img_path = entry.second[il];
                    if (params.verbose >= 2) {
                        printf("%s: processing image file '%s'\n", __func__, img_path.c_str());
                    }

                    if (!clip_image_load_from_file(img_path.c_str(), &img_inputs[il % batch_size])) {
                        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
                        continue;
                    }

                    image_file_index.push_back(img_path);
                }

                auto img_inputs_batch = clip_image_u8_batch_make(img_inputs);
                auto imgs_resized_batch = clip_image_f32_batch_make(imgs_resized);

                clip_image_batch_preprocess(clip_ctx, params.n_threads, &img_inputs_batch, &imgs_resized_batch);
                int64_t t0 = ggml_time_us();
                clip_image_batch_encode(clip_ctx, params.n_threads, &imgs_resized_batch, vec, true);
                int64_t t1 = ggml_time_us();
                if (params.verbose == 1) {
                    printf(".");
                }
                encode_timing_ms.push_back(((t1 - t0) / 1000.0) / leftover);

                // add image vectors to the database
                for (size_t l = 0; l < leftover; l++) {
                    alg_hnsw->addPoint(vec.data() + l * vec_dim, l);
                }
                printf("\n");
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

        printf("[%s]: Average encoding time / stddev: %.2fms - %.2f\n", __func__, mean, stddev);
    }

    // Serialize index and save the vectors
    std::string hnsw_path = params.vecdb;
    printf("[%s]: persist vector DB '%s'\n", __func__, params.vecdb.c_str());
    printf("[%s]: vector DB max_elements-[%zd] / cur_elements-[%zd]\n", __func__,
        alg_hnsw->getMaxElements(), alg_hnsw->getCurrentElementCount());
    alg_hnsw->saveIndex(params.vecdb);

    // Save filepaths matching vector DB for decoding query results
    std::ofstream image_file_index_file(params.filepaths.c_str(), std::ios::binary | std::ios::trunc);
    // first line is model
    image_file_index_file << params.model << "\n";
    for (const auto & i_path : image_file_index) {
        image_file_index_file << i_path << "\n";
    }

    printf("[%s]: %zu images processed and indexed\n", __func__, image_file_index.size());

    delete alg_hnsw;

    clip_free(clip_ctx);

    int64_t t_elapsed = ggml_time_us() - t_start;

    printf("\n[%s]: Elapsed time: %.2fs\n", __func__, t_elapsed / 1000.0 / 1000.0);
    print_tensor_op_perf_data(t_elapsed);

    // Run small test to verify the DB is valid
    testVecDb();

    return 0;
}
