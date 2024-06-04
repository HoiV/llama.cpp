#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...

#include "kv-cache.h"

llama_context *ctx;
llama_context_params ctx_params;
llama_model *model;
llama_model_params model_params;
int n_decode = 0;
bool save_slm_state = false;
std::vector<llama_token> session_tokens;
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
    model_params = llama_model_default_params();

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
        // start from a known point
        llama_kv_cache_clear(ctx);

        std::string template_prompt = params.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            params.pfx_shared = template_prompt.substr(0, pos);
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            tokens_shared = llama_tokenize(model, params.pfx_shared, false, false);

#if 1 // use llama_state_load_file()
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
                printf("%s: Load state file failed: %s\n", __func__, params.pfx_file.c_str());
                session_tokens.resize(0);
                save_slm_state = true;
                tokens_shared.clear();
                params.pfx_shared = "";

                // for now this plug-in should not create cache files - comment this out for cache generation
                return 1;
            }
            else {
                printf("%s: Loading saved state...\n", __func__);
                session_tokens.resize(n_token_count_out);
                llama_set_rng_seed(ctx, params.seed);
                // printf("%s: n_token_count_out=%zd: %s\n", __func__, n_token_count_out, LOG_TOKENS_TOSTR_PRETTY(ctx, session_tokens).c_str());

                // sanity check
                GGML_ASSERT(tokens_shared.size() <= session_tokens.size());
                for (size_t i = 0; i < tokens_shared.size(); i++) {
                    GGML_ASSERT(tokens_shared[i] == session_tokens[i]);
                    // printf("%d: %d - %d\n", i, tokens_shared[i], session_tokens[i]);
                    if (tokens_shared[i] != session_tokens[i]) {
                        printf("Mismatched pfx tokens!!!!\n");
                        return 1;
                    }
                }

                //printf("%s: token_shared=%zd - %s\n", __func__, tokens_shared.size(), LOG_TOKENS_TOSTR_PRETTY(ctx, tokens_shared).c_str());

                // remove any "future" tokens that we might have inherited from the previous session
                llama_kv_cache_seq_rm(ctx, -1, tokens_shared.size(), -1);
            }

#else // use llama_set_state_data()

            printf("%s: Loading saved state...\n", __func__);
            FILE *slm_state_fp = fopen("./slm_state.bin", "rb");
            if (slm_state_fp != NULL) {
                auto state_size = llama_state_get_size(ctx);
                //if (state_size != state_size2) {
                //    cerr << "state size differs\n";
                //}
                auto state_mem = new uint8_t[state_size];
                fread(state_mem, 1, state_size, slm_state_fp);
                llama_state_set_data(ctx, state_mem);  // could also read directly from memory mapped file
                fclose(slm_state_fp);
                printf("%s: saved state loaded successfully\n", __func__);
            } else {
                printf("%s: slm state file not available - set flag to save it later...\n", __func__);
                save_slm_state = true;
            }

#endif

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
    std::vector<llama_token> embd_inp;
    int n_consumed = 0;
    int n_past = 0;
    int n_kv_pfx = 0;

    if (params.pfc_mode) {
        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, tokens_shared.size(), -1);
        embd_inp.insert(embd_inp.end(), tokens_shared.begin(), tokens_shared.end());
        n_consumed = tokens_shared.size();
        n_past = tokens_shared.size();
        n_kv_pfx = tokens_shared.size();
    } else {
        n_consumed = 0;
        n_past = 0;
        n_kv_pfx = 0;
    }

    // tokenize the remaining prompt or full prompt if pfc_mode is off
    std::vector<llama_token> tokens_input = llama_tokenize(model, params.prompt, false, false);

    // append the variant part of the prompt or the full prompt for non pfc mode
    embd_inp.insert(embd_inp.end(), tokens_input.begin(), tokens_input.end());

    const int n_ctx = llama_n_ctx(ctx);
    const int n_kv_req = tokens_input.size() + (params.n_len - tokens_input.size() - n_kv_pfx);

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

    // printf("%s: before eval n_past=%d, eval: %s\n", __func__, n_past, LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // calculate how much has been processed through the saved state file
    int prompt_index = 0;
    if (params.pfc_mode) {
        int n_tokens_processed = 0;
        for (; prompt_index < embd_inp.size(); prompt_index++) {
            // not fully matched with shared tokens
            if (embd_inp[prompt_index] != tokens_shared[prompt_index]) {
                break;
            }

            n_tokens_processed++;

            // embd_inp is fully matched with shared prefix cache
            if (n_tokens_processed >= (int)tokens_shared.size()) {
                ++prompt_index;
                break;
            }
        }

        // printf("%s: pfc mode - tokens processed =%d - prompt_index=%d - n_past=%d\n", __func__, n_tokens_processed, prompt_index, n_past);
    }

    // build token list for inference
    std::vector<llama_token> embd;
    for (int i = prompt_index; i < embd_inp.size(); i++) {
        embd.push_back(embd_inp[i]);
    }
    // printf("%s: decode: %s\n", __func__, LOG_TOKENS_TOSTR_PRETTY(ctx, embd).c_str());

    printf("%s: start decoding @n_past = %d - inference size = %zd\n", __func__, n_past, embd.size());
    int64_t t1_start = ggml_time_us();

    // decode the remaining prompt not covered by the shared portion
    for (int i = 0; i < (int)embd.size(); i += params.n_batch) {
        int n_eval = (int) embd.size() - i;
        if (n_eval > params.n_batch) {
            n_eval = params.n_batch;
        }

        if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
            printf("%s : failed to eval\n", __func__);
            return 1;
        }

        n_past += n_eval;
        printf("%s: decoded %d tokens\n", __func__, n_eval);
    }

    int64_t t2_start = ggml_time_us();
    printf("t2-t1 = %.2fms\n", ((t2_start - t1_start) / 1000.0f));

    if (params.pfc_mode && save_slm_state) {
        session_tokens.insert(session_tokens.end(), embd_inp.begin(), embd_inp.end());
    }

    if (save_slm_state) {
#if 1 // use llama_state_load_file()
        GGML_ASSERT(!file_exists(params.pfx_file));
        llama_state_save_file(ctx, params.pfx_file.c_str(), session_tokens.data(), session_tokens.size());
        save_slm_state = false;

        // update token_shared
        std::string template_prompt = params.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            params.pfx_shared = template_prompt.substr(0, pos);
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            tokens_shared = llama_tokenize(model, params.pfx_shared, false, false);
        }

#else // use llama_set_state_data()
        // save state (rng, logits, embedding and kv_cache) to file
        printf("%s: saving SLM state...\n", __func__);
        FILE *slm_state_fp = fopen("./slm_state.bin", "wb");
        auto state_size = llama_state_get_size(ctx);
        auto state_mem = new uint8_t[state_size];
        llama_state_get_data(ctx, state_mem);
        fwrite(state_mem, 1, state_size, slm_state_fp);
        fclose(slm_state_fp);
        save_slm_state = false;
        printf("%s: DONE saving SLM state...\n", __func__);
#endif
    }

    // compute max_len output
    int max_len = std::min(params.n_len, (n_past + 128));

    std::string slm_output;
    bool valid_reply = false;

    while (n_past <= max_len) {
        // sample the last token just received
        {
            auto n_vocab = llama_n_vocab(model);
            auto *logits = llama_get_logits_ith(ctx, 0);

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
            if (llama_token_is_eog(model, new_token_id)) {
                break;
            }

            const std::string token_str = llama_token_to_piece(ctx, new_token_id);

            if (token_str.find('{') != std::string::npos) {
                // accepted answers have '{' characters
                valid_reply = true;
            }

            if (valid_reply) {
#if 0 // enable the following printf for streaming replies
            printf("%s", token_str.c_str());
#else
            slm_output += token_str;
#endif
            }

            if (token_str.find('}') != std::string::npos) {
                // force end of output since we have a valid JSON reply
                break;
            }

            // save this new token for next evaluation
            embd.clear();
            embd.push_back(new_token_id);

            n_decode += 1;
        }

        // bump current generated token index
        n_past += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, llama_batch_get_one(&embd[0], 1, n_past, 0))) {
            printf("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    // we have reached max_len of output, hit eog char or "}"
    if (!valid_reply) {
        // reply not correctly formatted or unhelpful
        printf("%s: ***** invalid formatted reply from model *****\n", __func__);
    }

    printf("%s\n", slm_output.c_str());
    slm_output.clear();
    valid_reply = false;
    fflush(stdout);

    int64_t t3_start = ggml_time_us();
    printf("t3-t2 = %.2fms\n", ((t3_start - t2_start) / 1000.0f));

    return 0;
}

int slm_infer(gpt_params& params) {
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

    std::string slm_output;

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

