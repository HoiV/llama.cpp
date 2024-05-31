#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...

#include "kv-cache.h"

llama_context *ctx;
llama_context_params ctx_params;
llama_model *model;
int n_decode = 0;
std::vector<llama_token> session_tokens;
std::string slm_output;
int64_t t_main_start;
std::vector<llama_token> tokens_shared;

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
           const std::string & text,
                        bool   add_special,
                        bool   parse_special = false) {
    // upper limit for the number of tokens
    int n_tokens = text.length() + 2 * add_special;
    std::vector<llama_token> result(n_tokens);
    n_tokens = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_tokenize(model, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        GGML_UNUSED(check);
        GGML_ASSERT(check == -n_tokens);
    }
    else {
        result.resize(n_tokens);
    }
    return result;
}

std::string llama_token_to_piece(const struct llama_context * ctx, llama_token token, bool special = true) {
    std::vector<char> result(8, 0);
    const int n_tokens = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), special);
    if (n_tokens < 0) {
        result.resize(-n_tokens);
        int check = llama_token_to_piece(llama_get_model(ctx), token, result.data(), result.size(), special);
        GGML_ASSERT(check == -n_tokens);
    } else {
        result.resize(n_tokens);
    }

    return std::string(result.data(), result.size());
}

void llama_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token[batch.n_tokens] = id;
    batch.pos[batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits[batch.n_tokens] = logits;
    batch.n_tokens++;
}

std::string pfx_file_path(std::string pfx) {
    static std::hash<std::string> hasher;
    static std::string dir = "./llama_cache";
    
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

int slm_init(gpt_params& params) {
    // init LLM
    llama_backend_init();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();

    model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        printf("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context
    ctx_params = llama_context_default_params();

    ctx_params.seed  = params.seed;
    ctx_params.n_ctx = params.n_ctx;
    ctx_params.n_batch = params.n_ctx;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;

    ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        printf("%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    printf("\n%s: n_len = %d, n_ctx = %d\n", __func__, params.n_len, llama_n_ctx(ctx));
    printf("%s: n_threads = %d, n_threads_batch = %d\n\n", __func__, ctx_params.n_threads, ctx_params.n_threads_batch);

    if (params.pfc_mode) {
        std::string full_prompt = params.custom_template_prompt;
        size_t pos = full_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            params.pfx_shared = full_prompt.substr(0, pos);
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            tokens_shared = llama_tokenize(model, params.pfx_shared, false, false);
            // build the cache file directory
            params.pfx_file = pfx_file_path(params.pfx_shared);
            // load the cache and create one if it does not exist
            session_tokens.resize(params.n_ctx);
            size_t n_token_count_out = params.first_prompt ? 0xffffffff : 0;
            if (!llama_state_load_file(ctx, 
                                       params.pfx_file.c_str(),
                                       session_tokens.data(),
                                       session_tokens.capacity(),
                                       &n_token_count_out)) {
                session_tokens.resize(0);
                return 1;
            }
            else {
                session_tokens.resize(n_token_count_out);
                llama_set_rng_seed(ctx, params.seed);

                // sanity check
                GGML_ASSERT(tokens_shared.size() <= session_tokens.size());
                for (size_t i = 0; i < tokens_shared.size(); i++) {
                    GGML_ASSERT(tokens_shared[i] == session_tokens[i]);
                }

                // remove any "future" tokens that we might have inherited from the previous session
                llama_kv_cache_seq_rm(ctx, -1, tokens_shared.size(), -1);
            }
        }
        else {
            // no shared prompt detected
            tokens_shared.clear();
        }
    }
    else {
        // No pfc mode
        tokens_shared.clear();
    }

    t_main_start = ggml_time_us();

    return 0;
}

int slm_inference(gpt_params& params) {
    const llama_model* model = llama_get_model(ctx);

    // for custom_prompt always clear the cache since we want 
    // every prompt to start from the same beginning
    llama_kv_cache_clear(ctx);

    int n_kv_pfx = tokens_shared.size();

    // tokenize the remaining prompt or full prompt if pfc_mode is off
    std::vector<llama_token> tokens_list;
    tokens_list = llama_tokenize(model, params.prompt, true);

    const int n_ctx = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (params.n_len - tokens_list.size() - n_kv_pfx);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        printf("%s: error: n_kv_req(%d-%d) > n_ctx(%d), the required KV cache size is not big enough\n",
            __func__,
            n_kv_pfx,
            n_kv_req,
            n_ctx);
        printf("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return 1;
    }

    // create a llama_batch with size n_len to submit token data for decoding
    llama_batch batch = llama_batch_init(params.n_len, 0, 1);

    // evaluate the prompt
    size_t batch_ofs = 0;
    if (params.pfc_mode) {
        batch_ofs = tokens_shared.size();
        // add the shared prompt
        for (size_t i = 0; i < tokens_shared.size(); i++) {
            llama_batch_add(batch, tokens_shared[i], i, { 0 }, false);
        }
    }
    // insert the variant part of the prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], (i + batch_ofs), { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        printf("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop
    int n_cur = batch.n_tokens;
    printf("> token generation begins - n_cur = %d\n", n_cur);

    while (n_cur <= params.n_len) {
        // sample the last token just received
        {
            auto n_vocab = llama_n_vocab(model);
            auto *logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);

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

            llama_sample_top_k(ctx, &candidates_p, top_k, 1);
            llama_sample_top_p(ctx, &candidates_p, top_p, 1);
            llama_sample_temp(ctx, &candidates_p, temp);

            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation - are we done?
            if (llama_token_is_eog(model, new_token_id) || (n_cur > params.n_len)) {
                printf("\n");
                break;
            }

            const std::string token_str = llama_token_to_piece(ctx, new_token_id);
            // printf("%s", token_str.c_str());
            slm_output += token_str;
            fflush(stdout);

            if (token_str.c_str()[0] == '}') {
                // force end of output since we have what we need
                printf("%s", slm_output.c_str());
                slm_output.clear();
                printf("\n");
                break;
            }

            // prepare the next batch
            batch.n_tokens = 0;

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        // bump current generated token index
        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            printf("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    llama_batch_free(batch);

    return 0;
}

void slm_terminate() {
    printf("\n");

    int64_t t_main_end = ggml_time_us();

    printf("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);
    print_tensor_op_perf_data();
    printf("\n");

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();
}

