#pragma warning (disable:4100) // unref formal param
#pragma warning (disable:4245) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4267) // conversion from 'int64_t' to 'int'
#pragma warning (disable:4505) // unref function with internal linkage removed

#include "llm-infer.h"

#include "llama.h"
#include "common.h"
#include "log.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include "json.hpp" // For JSON handling

// For SLM 
llama_model *llm_model;
llama_context *llm_ctx;
std::vector<llama_token> llm_session_tokens;
std::vector<llama_token> llm_tokens_shared;

// For embeddings
llama_model * embed_model;
llama_context * embed_ctx;


void default_log_callback(
    ggml_log_level level, 
    const char * text, 
    void * user_data) {
    GGML_UNUSED(text);

    ggml_log_level llindex_log_level = (ggml_log_level)0 /* GGML_LOG_LEVEL_NONE */;
    if (user_data != nullptr) {
        llindex_log_level = *(ggml_log_level *)user_data;
    }

    if (level == llindex_log_level) {
        fputs(text, stdout);
    }
}

std::string pfx_file_path(
    std::string pfx) {

    static std::hash<std::string> hasher;
    static std::string dir = "./ggml_cache";
    
    // create the cache dir if it does not exist yet
    if (!CreateDirectoryA(dir.c_str(), NULL)) {
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            fprintf(stderr, "%s: Failed to create directory: %s - use current dir for prefix cache\n",
                __func__, dir.c_str());
            dir = ".";
        }
    }

    // default generated file name
    std::string full_file_path = dir + "/" + std::to_string(hasher(pfx));

    return full_file_path;
}

LLM_INFER_API
bool llm_initialize(
    model_params & params) {

    // init Llama backend
    llama_backend_init();

    // Control the default verbosity for llama.cpp
    llama_log_set(default_log_callback, &(params.verbose));

    // initialize the model
    llama_model_params common_model_params = llama_model_default_params();

#ifdef GGML_USE_CUDA
    #pragma message("++++++++ Support both CUDA and CPU")
    if ((ggml_backend_cuda_get_device_count() != 0) && (params.force_cpu_mode == 0)) {
        // if there is a GPU then make use of it
        common_model_params.n_gpu_layers = 999;
    } else {
        // either there is no GPU or no CPU forcing function
        common_model_params.n_gpu_layers = 0;
    }
#endif // GGML_USE_CUDA

    llm_model = llama_load_model_from_file(params.model_name.c_str(), common_model_params);
    if (llm_model == NULL) {
        printf("%s: error: unable to load model\n" , __func__);
        return false;
    }

    // initialize the context
    llama_context_params llm_ctx_params = llama_context_default_params();

    // Grab values from defaults or command line args 
    llm_ctx_params.seed  = params.seed;
    llm_ctx_params.n_ctx = params.n_ctx;
    llm_ctx_params.n_batch = params.n_batch;
    llm_ctx_params.n_threads = params.n_threads;
    llm_ctx_params.n_threads_batch = params.n_threads;

    llm_ctx = llama_new_context_with_model(llm_model, llm_ctx_params);

    if (llm_ctx == NULL) {
        printf("%s: error: failed to create the llama_context\n" , __func__);
        return false;
    }

    printf("\n%s: n_len = %d, n_ctx = %d\n", __func__, params.n_len, llama_n_ctx(llm_ctx));
    printf("%s: n_threads = %d, n_threads_batch = %d\n\n", __func__, llm_ctx_params.n_threads, llm_ctx_params.n_threads_batch);

    std::vector<llama_token> llm_tokens_shared;
    if (params.pfc_mode) {
        // start from a known point
        llama_kv_cache_clear(llm_ctx);

        std::string template_prompt = params.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            params.pfx_shared = template_prompt.substr(0, pos);
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            llm_tokens_shared = llama_tokenize(llm_model, params.pfx_shared, false, false);
            // build the cache file directory
            params.pfx_file = pfx_file_path(params.pfx_shared);
            // load the cache and create one if it does not exist
            llm_session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = params.first_prompt ? 0xffffffff : 0;
            if (!llama_state_load_file(llm_ctx, 
                                       params.pfx_file.c_str(),
                                       llm_session_tokens.data(),
                                       llm_session_tokens.capacity(),
                                       &n_token_count_out)) {
                printf("%s: Load state file failed: %s\n", __func__, params.pfx_file.c_str());
                llm_session_tokens.resize(0);
                params.save_llm_state = true;
                llm_tokens_shared.clear();
                params.pfx_shared = "";

                // for now this plug-in should not create cache files - comment this out for cache generation
                // return false;
            }
            else {
                printf("%s: Loading saved state from '%s'...\n", __func__, params.pfx_file.c_str());
                llm_session_tokens.resize(n_token_count_out);
                llama_set_rng_seed(llm_ctx, params.seed);
                // printf("%s: n_token_count_out=%zd: %s\n", __func__, n_token_count_out, LOG_TOKENS_TOSTR_PRETTY(ctx, session_tokens).c_str());

                // sanity check
                // assert(llm_tokens_shared.size() <= llm_session_tokens.size());
                for (size_t i = 0; i < llm_tokens_shared.size(); i++) {
                    // assert(llm_tokens_shared[i] == llm_session_tokens[i]);
                    if (llm_tokens_shared[i] != llm_session_tokens[i]) {
                        printf("Mismatched pfx tokens!!!!\n");
                        return false;
                    }
                }

                //printf("%s: token_shared=%zd\n", __func__, llm_tokens_shared.size());

                // remove any "future" tokens that we might have inherited from the previous session
                llama_kv_cache_seq_rm(llm_ctx, -1, llm_tokens_shared.size(), -1);
            }
        }
        else {
            // no shared prompt detected
            llm_tokens_shared.clear();
        }
    }
    else {
        // No pfc mode
        llm_tokens_shared.clear();
    }

    return true;
}

LLM_INFER_API
bool llm_inference(
    model_params& params) {

    std::vector<llama_token> embd_inp;
    int n_consumed = 0;
    int n_past = 0;
    int n_kv_pfx = 0;

    std::vector<llama_token> llm_tokens_shared = llm_tokens_shared;
    if (params.pfc_mode) {
        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(llm_ctx, -1, llm_tokens_shared.size(), -1);
        embd_inp.insert(embd_inp.end(), llm_tokens_shared.begin(), llm_tokens_shared.end());
        n_consumed = llm_tokens_shared.size();
        n_past = llm_tokens_shared.size();
        n_kv_pfx = llm_tokens_shared.size();
    } else {
        // start from a known point
        llama_kv_cache_clear(llm_ctx);

        n_consumed = 0;
        n_past = 0;
        n_kv_pfx = 0;
    }

    // tokenize the remaining prompt or full prompt if pfc_mode is off
    std::vector<llama_token> tokens_input = llama_tokenize(llm_model, params.prompt, false, false);

    // append the variant part of the prompt or use the full prompt (for non pfc mode)
    embd_inp.insert(embd_inp.end(), tokens_input.begin(), tokens_input.end());

    const int n_ctx = llama_n_ctx(llm_ctx);
    const int n_kv_req = tokens_input.size() + (params.n_len - tokens_input.size() - n_kv_pfx);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        printf("%s: error: n_kv_req(%d-%d) > n_ctx(%d), the required KV cache size is not big enough\n",
            __func__,
            n_kv_pfx,
            n_kv_req,
            n_ctx);
        printf("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return false;
    }

    // calculate how much has been processed through the saved state file
    int prompt_index = 0;
    if (params.pfc_mode) {
        int n_tokens_processed = 0;
        for (; prompt_index < embd_inp.size(); prompt_index++) {
            // not fully matched with shared tokens
            if (embd_inp[prompt_index] != llm_tokens_shared[prompt_index]) {
                break;
            }

            n_tokens_processed++;

            // embd_inp is fully matched with shared prefix cache
            if (n_tokens_processed >= (int)llm_tokens_shared.size()) {
                ++prompt_index;
                break;
            }
        }
    }

    // build token list for inference
    std::vector<llama_token> embd;
    for (int i = prompt_index; i < embd_inp.size(); i++) {
        embd.push_back(embd_inp[i]);
    }

    int64_t t1_start = ggml_time_us();
    GGML_UNUSED(t1_start );

    // decode the remaining prompt not covered by the shared portion
    for (int i = 0; i < (int)embd.size(); i += params.n_batch) {
        int n_eval = (int) embd.size() - i;
        if (n_eval > params.n_batch) {
            n_eval = params.n_batch;
        }

        if (llama_decode(llm_ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
            printf("%s : failed to eval\n", __func__);
            return false;
        }

        // update n_past to reflect what has been decoded
        n_past += n_eval;
    }

    int64_t t2_start = ggml_time_us();
    if (params.verbose != 0) {
        printf("Prompt eval time = %.2fms\n", ((t2_start - t1_start) / 1000.0f));
    }

    if (params.pfc_mode && params.save_llm_state) {
        llm_session_tokens.insert(llm_session_tokens.end(), embd_inp.begin(), embd_inp.end());
    }

    if (params.save_llm_state) {
        llama_state_save_file(llm_ctx, params.pfx_file.c_str(), llm_session_tokens.data(), llm_session_tokens.size());
        params.save_llm_state = false;

        // update token_shared
        std::string template_prompt = params.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            params.pfx_shared = template_prompt.substr(0, pos);
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            llm_tokens_shared = llama_tokenize(llm_model, params.pfx_shared, false, false);
        }
    }

    // compute max_len output
    int max_len = std::min(params.n_len, (n_past + 128));

    std::string slm_output;
    int n_tokens_generated = 0;

    while (n_past <= max_len) {
        // sample the last token just received
        {
            auto n_vocab = llama_n_vocab(llm_model);
            auto *logits = llama_get_logits_ith(llm_ctx, 0);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token (greedy sampling algo)
            const int   top_k = 40;
            const float top_p = 0.9f;
            const float temp = 0.1f;

            llama_sample_top_k(llm_ctx, &candidates_p, top_k, 1);
            llama_sample_top_p(llm_ctx, &candidates_p, top_p, 1);
            llama_sample_temp(llm_ctx, &candidates_p, temp);

            const llama_token new_token_id = llama_sample_token_greedy(llm_ctx, &candidates_p);

            // we reached the end of token generation
            if (llama_token_is_eog(llm_model, new_token_id)) {
                printf("\n");
                break;
            }

            const std::string token_str = llama_token_to_piece(llm_ctx, new_token_id);
            if (params.streaming_reply) {
                printf("%s", token_str.c_str());
            } else {
                slm_output += token_str;
            }

            // save this new token for next evaluation
            embd.clear();
            embd.push_back(new_token_id);

            n_tokens_generated += 1;
            params.total_llm_tokens_generated += 1;
        }

        // bump current generated token index
        n_past += 1;

        // decode the output for the new generated token
        if (llama_decode(llm_ctx, llama_batch_get_one(&embd[0], 1, n_past, 0))) {
            printf("%s : failed to eval, return code %d\n", __func__, 1);
            return false;
        }
    }

#if 0
    int64_t t_us = (ggml_time_us() - t2_start);
    printf("> token generation time = %.2fms (%d) (%.2ft/s) (%.2fms)\n", 
        t_us / 1000.0f,
        n_tokens_generated, 
        n_tokens_generated / (t_us / 1000000.0f),
        (t_us / (n_tokens_generated * 1000.0f)));

    // accumulate the token generation time
    params.t_llm_token_generation_time += t_us;
#endif

    params.reply = slm_output;

    return true;
}

LLM_INFER_API 
void llm_terminate(const model_params& ) {

    int verbose = GGML_LOG_LEVEL_INFO;
    llama_log_set(default_log_callback, &verbose);
    llama_print_timings(llm_ctx);

    llama_free(llm_ctx);
    llama_free_model(llm_model);

    llama_backend_free();
}

static void batch_decode(
    llama_context * ctx, 
    llama_batch & batch, 
    float * output, 
    int /* n_seq */, 
    int n_embd) {

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

LLM_INFER_API
bool embed_initialize(
    model_params & params) {

    llama_backend_init();

    llama_log_set(default_log_callback, &(params.verbose));

    // initialize the model
   llama_model_params llama_model_params = llama_model_default_params();

   embed_model = llama_load_model_from_file(params.model_name.c_str(), llama_model_params);
   if (embed_model == NULL) {
       printf("%s: error: unable to load model\n" , __func__);
       return false;
   }

   // initialize the context
   llama_context_params embed_ctx_params = llama_context_default_params();

   // Grab default params or values from command line args
   embed_ctx_params.seed  = params.seed;
   embed_ctx_params.n_ctx = params.n_ctx;
   embed_ctx_params.n_batch = params.n_batch;
   embed_ctx_params.n_threads = params.n_threads;
   embed_ctx_params.n_threads_batch = params.n_threads;

   // For BERT models batch size must be equal to ubatch size
   embed_ctx_params.n_ubatch = params.n_batch;
   embed_ctx_params.embeddings = true;

   embed_ctx = llama_new_context_with_model(embed_model, embed_ctx_params);

   if (embed_ctx == NULL) {
       printf("%s: error: failed to create the llama_context\n" , __func__);
       return false;
   }

   // Save dimension from model hparams
   params.n_dim = llama_n_embd(embed_model);

   const int n_ctx_train = llama_n_ctx_train(embed_model);
    const int n_ctx = llama_n_ctx(embed_ctx);

    const enum llama_pooling_type pooling_type = llama_pooling_type(embed_ctx);
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        fprintf(stderr, "%s: error: pooling type NONE not supported\n", __func__);
        return false;
    }

    if (n_ctx > n_ctx_train) {
        fprintf(stderr, "%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
        return false;
    }

    return true;
}

LLM_INFER_API
bool embed_encode_batch_v1(
    const model_params & params, 
    std::vector<chunk> & chunks) {

    const size_t n_chunks = chunks.size();
    struct llama_batch batch = llama_batch_init(params.n_batch, 0, 1);

    // Allocate output
    int n_embd = llama_n_embd(embed_model);

    // Tokenize the prompts and trim
    for (auto & chunk : chunks) {
        auto inp = llama_tokenize(embed_ctx, chunk.textdata, true, false);
        if (inp.size() > params.n_batch) {
            fprintf(stderr, "%s: error: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) inp.size(), (long long int) params.n_batch);
            return false;
        }

        // add eos if not present
        if (llama_token_eos(embed_model) >= 0 && (inp.empty() || inp.back() != llama_token_eos(embed_model))) {
            inp.push_back(llama_token_eos(embed_model));
        }

        chunk.tokens = inp;
    }

    // Break into batches
    int p = 0; // number of prompts already processed
    int s = 0; // number of prompts in current batch

    std::vector<float> embeddings(n_chunks * n_embd, 0);
    float * emb = embeddings.data();

    for (int k = 0; k < n_chunks; k++) {
        // Clamp to n_batch tokens
        auto & rag_tokens = chunks[k].tokens;
        const uint64_t n_toks = rag_tokens.size();

        // Encode if at capacity
        if (batch.n_tokens + n_toks > params.n_batch) {
            float * out = emb + p * n_embd;
            batch_decode(embed_ctx, batch, out, s, n_embd);
            llama_batch_clear(batch);
            p += s;
            s = 0;
        }

        // Add to batch
        size_t n_tokens = rag_tokens.size();
        for (size_t i = 0; i < n_tokens; i++) {
            llama_batch_add(batch, rag_tokens[i], (llama_pos) i, { s }, true);
        }

        s += 1;
    }

    // Final batch
    float * out = emb + p * n_embd;
    batch_decode(embed_ctx, batch, out, s, n_embd);

    // Save embeddings to chunks in memory
    for (int i = 0; i < n_chunks; i++) {
        chunks[i].embeddings = std::vector<float>(emb + i * n_embd, emb + (i + 1) * n_embd);
    }

    llama_batch_free(batch);

    return true;
}

LLM_INFER_API
bool embed_encode_batch(
    const model_params & params, 
    std::vector<chunk> & chunks) {

    const size_t n_chunks = chunks.size();
    struct llama_batch batch = llama_batch_init(params.n_batch, 0, 1);

    // Allocate output
    int n_embd = llama_n_embd(embed_model);
    std::vector<float> chunk_embeddings(n_embd, 0);

    // Tokenize the prompts and trim
    for (auto & chunk : chunks) {
        chunk.tokens = llama_tokenize(embed_ctx, chunk.textdata, true, false);
        size_t n_tokens = chunk.tokens.size();
        if (n_tokens > params.n_batch) {
            fprintf(stderr, "%s: error: chunk size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                    __func__, (long long int) n_tokens, (long long int) params.n_batch);
            return false;
        }

        // add eos if not present
        if (llama_token_eos(embed_model) >= 0 && (chunk.tokens.empty() || chunk.tokens.back() != llama_token_eos(embed_model))) {
            chunk.tokens.push_back(llama_token_eos(embed_model));
        }


        // Add to batch
        for (size_t i = 0; i < n_tokens; i++) {
            llama_batch_add(batch, chunk.tokens[i], (llama_pos) i, { 0 }, true);
        }

        batch_decode(embed_ctx, batch, chunk_embeddings.data(), 1, n_embd);
        chunk.embeddings = chunk_embeddings;

        llama_batch_clear(batch);
    }

    llama_batch_free(batch);

    return true;
}

LLM_INFER_API
bool embed_encode_single(
    const model_params & params,
    const std::string& query,
    std::vector<float> & embeddings) {

    const int n_embd = llama_n_embd(embed_model);
    std::vector<float> query_embeddings(n_embd, 0);
    
    std::vector<int32_t> query_tokens = llama_tokenize(embed_ctx, query, true);
    size_t n_tokens = query_tokens.size();
    if (n_tokens > params.n_batch) {
        fprintf(stderr, "%s: error: query string size (%lld) exceeds batch size (%lld), increase batch size and re-run\n",
                __func__, (long long int) n_tokens, (long long int) params.n_batch);
        return false;
    }

    // add eos if not present
    if ((llama_token_eos(embed_model) >= 0) && 
        (query_tokens.empty() || (query_tokens.back() != llama_token_eos(embed_model)))) {
        query_tokens.push_back(llama_token_eos(embed_model));
    }

    
    struct llama_batch embed_query_batch = llama_batch_init(params.n_batch, 0, 1);
    for (size_t i = 0; i < n_tokens; i++) {
        llama_batch_add(embed_query_batch, query_tokens[i], (llama_pos) i, { 0 }, true);
    }

    // query_embeddings contains the result from sentencepiece (all-MiniLM-L6-v2)
    batch_decode(embed_ctx, embed_query_batch, query_embeddings.data(), 1, n_embd);
    embeddings = query_embeddings;

    llama_batch_clear(embed_query_batch);
    llama_batch_free(embed_query_batch);

    return true;
}

LLM_INFER_API
void embed_terminate() {
    llama_free(embed_ctx);
    llama_free_model(embed_model);

    // This step is done via the llm_terminate() side
    // llama_backend_free();
}
