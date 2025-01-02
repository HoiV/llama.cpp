#include "common.h"
#include "llama.h"

#include <algorithm>
#include <fstream>

/* Retrieval */

void llindex_log_callback(ggml_log_level level, const char * text, void * user_data) {
    GGML_UNUSED(text);

    ggml_log_level llindex_log_level = (ggml_log_level)0 /* GGML_LOG_LEVEL_NONE */;
    if (user_data != nullptr) {
        llindex_log_level = *(ggml_log_level *)user_data;
    }

    if (level == llindex_log_level) {
        fputs(text, stdout);
    }
}

static void print_usage(int argc, char ** argv, const gpt_params & params) {
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s --model ./models/bge-base-en-v1.5-f16.gguf --context-file quotes.txt --chunk-size 1\n", argv[0]);
    LOG_TEE("\n");
}

struct chunk {
    // filename
    std::string filename;
    // original file position
    size_t filepos;
    // original text data
    std::string textdata = "";
    // tokenized text data
    std::vector<llama_token> tokens;
    // embedding
    std::vector<float> embedding;
};

// chunk file data to chunks of size >= chunk_size
// chunk_separator is the separator between chunks
static std::vector<chunk> chunk_file(const std::string & filename, int chunk_size, const std::string & chunk_separator) {
    std::vector<chunk> chunks;
    std::ifstream f(filename.c_str());

    if (!f.is_open()) {
        fprintf(stderr, "Error: could not open file %s\n", filename.c_str());
        return chunks;
    }

    chunk current_chunk;
    char buffer[1024];
    int64_t filepos = 0;
    std::string current = "";
    while (f.read(buffer, 1024)) {
        current += std::string(buffer, f.gcount());
        size_t pos;
        while ((pos = current.find(chunk_separator)) != std::string::npos) {
            current_chunk.textdata += current.substr(0, pos + chunk_separator.size());
            if ((int) current_chunk.textdata.size() > chunk_size) {
                // save chunk
                current_chunk.filepos = filepos;
                current_chunk.filename = filename;
                chunks.push_back(current_chunk);
                // update filepos
                filepos += (int) current_chunk.textdata.size();
                // reset current_chunk
                current_chunk = chunk();
            }
            current = current.substr(pos + chunk_separator.size());
        }
    }
    // add leftover data to last chunk
    if (current_chunk.textdata.size() > 0) {
        if (chunks.empty()) {
            current_chunk.filepos = filepos;
            current_chunk.filename = filename;
            chunks.push_back(current_chunk);
        } else {
            chunks.back().textdata += current_chunk.textdata;
        }
    }
    // if there is more data left then add it in as last chunk
    if (current.size() > 0) {
        current_chunk.filepos = filepos;
        current_chunk.filename = filename;
        current_chunk.textdata += current;
        chunks.push_back(current_chunk);
    }
    f.close();
    return chunks;
}

static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        llama_batch_add(batch, tokens[i], (llama_pos) i, { seq_id }, true);
    }
}

static void batch_decode(llama_context * ctx, llama_batch & batch, float * output, int /* n_seq */, int n_embd) {
    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    // fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        // try to get sequence embeddings - supported only when pooling_type is not NONE
        const float * embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
        if (embd == NULL) {
            embd = llama_get_embeddings_ith(ctx, i);
            if (embd == NULL) {
                fprintf(stderr, "%s: failed to get embeddings for token %d\n", __func__, i);
                continue;
            }
        }

        float * out = output + batch.seq_id[i][0] * n_embd;
        llama_embd_normalize(embd, out, n_embd);
    }
}

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv, params);
        return 1;
    }

    llama_log_set(llindex_log_callback, &(params.verbosity));

    // For BERT models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;
    params.embedding = true;

    if (params.chunk_size <= 0) {
        fprintf(stderr, "chunk_size must be positive\n");
        return 1;
    }
    if (params.context_files.empty()) {
        fprintf(stderr, "context_files must be specified\n");
        return 1;
    }

    int64_t t_main_start = ggml_time_us();

    printf("processing files: ");
    for (auto & context_file : params.context_files) {
        printf("%s\n", context_file.c_str());
    }

    std::vector<chunk> chunks;
    for (auto & context_file : params.context_files) {
        std::vector<chunk> file_chunk = chunk_file(context_file, params.chunk_size, params.chunk_separator);
        chunks.insert(chunks.end(), file_chunk.begin(), file_chunk.end());
    }
    printf("Number of chunks: %lld\n", chunks.size());

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model;
    llama_context * ctx;

    // load the model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        fprintf(stderr, "%s: error: pooling type NONE not supported\n", __func__);
        return 1;
    }

    if (n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    if (params.verbosity)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s\n", gpt_params_get_system_info(params).c_str());
    }

    // max batch size
    const uint32_t n_batch = params.n_batch;
    GGML_ASSERT(params.n_batch >= params.n_ctx);

    printf("%s: Tokenizing data...\n", __func__);
    int64_t t_tokenization_start = ggml_time_us();

    // tokenize the prompts and trim
    for (auto & chunk : chunks) {
        auto inp = ::llama_tokenize(ctx, chunk.textdata, true, false);
        if (inp.size() > n_batch) {
            fprintf(stderr, "%s: error: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) n_batch);
            return 1;
        }
        // add eos if not present
        if (llama_token_eos(model) >= 0 && (inp.empty() || inp.back() != llama_token_eos(model))) {
            inp.push_back(llama_token_eos(model));
        }
        chunk.tokens = inp;
    }

    int64_t t_tokenization_stop = ggml_time_us();

    // tokenization stats
    if (params.verbose_prompt) {
        for (int i = 0; i < (int) chunks.size(); i++) {
            fprintf(stderr, "%s: prompt %d: '%s'\n", __func__, i, chunks[i].textdata.c_str());
            fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, chunks[i].tokens.size());
            for (int j = 0; j < (int) chunks[i].tokens.size(); j++) {
                fprintf(stderr, "%6d -> '%s'\n", chunks[i].tokens[j], llama_token_to_piece(ctx, chunks[i].tokens[j]).c_str());
            }
            fprintf(stderr, "\n\n");
        }
    }

    printf("%s: Creating Embeddings...\n", __func__);
    int64_t t_embeddings_start = ggml_time_us();

    // initialize batch
    const size_t n_chunks = chunks.size();
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);

    // allocate output
    const int n_embd = llama_n_embd(model);
    std::vector<float> embeddings(n_chunks * n_embd, 0);
    float * emb = embeddings.data();

    // break into batches
    int p = 0; // number of prompts processed already
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_chunks; k++) {
        // clamp to n_batch tokens
        auto & inp = chunks[k].tokens;

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > n_batch) {
            float * out = emb + p * n_embd;
            batch_decode(ctx, batch, out, s, n_embd);
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }

        if ((k % 50) == 0) {
            printf("- Processing %d/%zd items\r", k, n_chunks);
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch
    float * out = emb + p * n_embd;
    batch_decode(ctx, batch, out, s, n_embd);

    int64_t t_embeddings_stop = ggml_time_us();

    // save embeddings to chunks
    for (int i = 0; i < n_chunks; i++) {
        chunks[i].embedding = std::vector<float>(emb + i * n_embd, emb + (i + 1) * n_embd);
        // clear tokens as they are no longer needed
        chunks[i].tokens.clear();
    }

    struct llama_batch query_batch = llama_batch_init(n_batch, 0, 1);

    printf("%s: Querying loop starts...\n", __func__);
    // start loop, read each query and return top-k or top similar chunk(s) 
    // based on cosine similarity
    int errors = 0;
    int item_count = 0;
    for (auto & context_file : params.context_files) {
        std::ifstream cpfile(context_file);
        if (!cpfile.is_open()) {
            printf("[%s]: failed to open [%s]\n", __func__, context_file.c_str());
            return false;
        }

        std::string query;
        int64_t t_query_start = ggml_time_us();

        while (std::getline(cpfile, query)) {
            std::vector<int32_t> query_tokens = llama_tokenize(ctx, query, true);

            batch_add_seq(query_batch, query_tokens, 0);

            std::vector<float> query_emb(n_embd, 0);
            batch_decode(ctx, query_batch, query_emb.data(), 1, n_embd);

            llama_batch_clear(query_batch);

            // compute cosine similarities
            {
                std::vector<std::pair<int, float>> similarities;
                for (int i = 0; i < n_chunks; i++) {
                    float sim = llama_embd_similarity_cos(chunks[i].embedding.data(), query_emb.data(), n_embd);
                    similarities.push_back(std::make_pair(i, sim));
                }

                // sort similarities
                std::sort(similarities.begin(), similarities.end(), [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
                    return a.second > b.second;
                });

#if 0
                // print top candidate
                printf("query: %s\n", query.c_str());
                printf("filename: %s\n", chunks[similarities[0].first].filename.c_str());
                printf("filepos: %lld\n", (long long int) chunks[similarities[0].first].filepos);
                printf("similarity: %f\n", similarities[0].second);
                printf("textdata:\n%s\n", chunks[similarities[0].first].textdata.c_str());            
#endif
                if ((chunks[similarities[0].first].textdata.find(query) == std::string::npos) &&
                    (query.find(chunks[similarities[0].first].textdata) == std::string::npos)) {
                    if (params.verbosity) {
                        printf("ERROR encountered on the following item: \n"
                            "s1 = [%s]\n"
                            "s2 = [%s]\n", 
                            query.c_str(),
                            chunks[similarities[0].first].textdata.c_str());
                    }
                    errors++;
                }
                if ((++item_count % 50) == 0) {
                    printf("- Processed %d items\r", item_count);
                }
            }
        }

        int64_t t_query_stop = ggml_time_us();
        printf("Total items processed: %d\n", item_count);
        printf("Query time             = %6.2fs (%5.2fms per item)\n", 
            (t_query_stop - t_query_start) / (1000.0 * 1000.0), 
            (t_query_stop - t_query_start) / (item_count * 1000.0));
    }

    printf("Tokenization time      = %6.2fms(%5.2fms per chunk)\n", 
        (t_tokenization_stop - t_tokenization_start) / 1000.0, 
        (t_tokenization_stop - t_tokenization_start) / (chunks.size() * 1000.0));
    printf("Create Embeddings time = %6.2fs (%5.2fms per chunk)\n", 
        (t_embeddings_stop - t_embeddings_start) / (1000.0 * 1000.0), 
        (t_embeddings_stop - t_embeddings_start) / (chunks.size() * 1000.0));
    printf("Errors                 = %3d\n", errors);

    params.verbosity = GGML_LOG_LEVEL_INFO;
    llama_log_set(llindex_log_callback, &(params.verbosity));
    llama_print_timings(ctx);

    // clean up
    llama_batch_free(query_batch);
    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();

    const auto t_main_end = ggml_time_us();
    printf("\n\ntotal elapsed time %7.2fsec\n\n", (double)(t_main_end - t_main_start) / (1000. * 1000.)); 

#ifdef GGML_TENSOR_OP_PERF
    print_tensor_op_perf_data(t_main_end - t_main_start);
#endif // GGML_TENSOR_OP_PERF
}
