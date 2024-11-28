#include "slm.h"

llama_context *ctx;
llama_context_params ctx_params;
llama_model *model;
llama_model_params model_params;
static int total_tokens_generated = 0;
std::vector<llama_token> session_tokens;
static int64_t t_token_generation = 0;
std::vector<llama_token> tokens_shared;
std::vector<std::string> custom_prompts;
xbapp_params xbparams;

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
        GGML_UNUSED(check);
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

bool processCustomPromptsFromFile() {
    std::ifstream cpfile(xbparams.custom_p_file);
    if (!cpfile.is_open()) {
        printf("[%s]: failed to open [%s]\n", __func__, xbparams.custom_p_file.c_str());
        return false;
    }

    std::string line;
    bool templatePromptMode = false;
    bool userPromptMode = false;
    custom_prompts.clear();
    std::string custom_template_prompt = "";

    // process CUSTOM_TEMPLATE_PROMPT
    while (std::getline(cpfile, line)) {
        if (line.find("CUSTOM_TEMPLATE_PROMPT") != std::string::npos) {
            templatePromptMode = true;
            continue;
        } else if (line.find("CUSTOM_PROMPT") != std::string::npos) {
            userPromptMode = true;
            continue;
        } else if (line.find("END_SECTION") != std::string::npos) {
            templatePromptMode = false;
            userPromptMode = false;
            continue;
        }

        if (templatePromptMode) {
            custom_template_prompt += line + '\n';
        } else if (userPromptMode) {
            custom_prompts.push_back(line);
        }
    }

    xbparams.custom_template_prompt = custom_template_prompt;

    cpfile.close();

    return true;
}

void xb_set_process_affinity(int64_t affinity_mask) {
    if (!SetProcessAffinityMask(GetCurrentProcess(), affinity_mask)) {
        printf("failed to set process group affinity 0x%08llx\n", affinity_mask);
    }
}

int slm_init() {
    CPUInfo cinfo;
    cout << "CPU vendor = " << cinfo.vendor() << endl;
    cout << "CPU Brand String = " << cinfo.model() << endl;
    cout << "# of cores = " << cinfo.cores() << endl;
    cout << "# of logical cores = " << cinfo.logicalCpus() << endl;
    cout << "Is CPU Hyper threaded = " << cinfo.isHyperThreaded() << endl;

    if (cinfo.vendor().find("AMD") != std::string::npos) {
        if (cinfo.model().find("AMD Ryzen AI 9 HX 370") != std::string::npos) {
            //printf("%s: Detected AMD Ryzen HX 370\n", __func__);
            xbparams.is_AMD_Ryzen_HX_370 = true;
        } else if (cinfo.model().find("AMD RYZEN AI MAX+ PRO 395") != std::string::npos) {
            //printf("%s: Detected AMD Ryzen PRO 395\n", __func__);
            xbparams.is_AMD_Ryzen_PRO_395 = true;
        }
    }

    printf("%s: Actual using: %d threads\n", __func__, xbparams.n_threads);

    // init LLM
    llama_backend_init();

    if (xbparams.openmp) {
        ggml_select_omp();
    }

    printf("[%s]: processing cpf input file [%s]\n", __func__, xbparams.custom_p_file.c_str());
    processCustomPromptsFromFile();

    // initialize the model
    model_params = llama_model_default_params();
    model_params.n_gpu_layers = xbparams.n_ngl;

    model = llama_load_model_from_file(xbparams.model_path.c_str(), model_params);
    if (model == NULL) {
        printf("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context
    ctx_params = llama_context_default_params();

    ctx_params.seed  = xbparams.seed;
    ctx_params.n_ctx = xbparams.n_ctx;
    ctx_params.n_batch = xbparams.n_ctx;
    ctx_params.n_threads = xbparams.n_threads;
    ctx_params.n_threads_batch = xbparams.n_threads;

    ctx = llama_new_context_with_model(model, ctx_params);
    if (ctx == NULL) {
        printf("%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    printf("\n%s: n_len = %d, n_ctx = %d\n", __func__, xbparams.n_len, llama_n_ctx(ctx));
    printf("%s: n_threads = %d, n_threads_batch = %d\n\n", __func__, ctx_params.n_threads, ctx_params.n_threads_batch);

    if (xbparams.pfc_mode) {
        // start from a known point
        llama_kv_cache_clear(ctx);

        std::string template_prompt = xbparams.custom_template_prompt;
        size_t pos = template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared prompt
            xbparams.pfx_shared = ::trim(template_prompt.substr(0, pos));
            // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
            tokens_shared = llama_tokenize(model, xbparams.pfx_shared, false, false);

            // load the cache and create one if it does not exist
            session_tokens.resize(xbparams.n_ctx);
            size_t n_token_count_out = xbparams.first_prompt ? 0xffffffff : 0;
            if (llama_state_load_file(ctx, 
                                      xbparams.pfx_file.c_str(),
                                      session_tokens.data(),
                                      session_tokens.capacity(),
                                      &n_token_count_out)) {

                printf("%s: Loading saved state from '%s' (size %zd)...\n", __func__, xbparams.pfx_file.c_str(), tokens_shared.size());
                session_tokens.resize(n_token_count_out);
                llama_set_rng_seed(ctx, xbparams.seed);
                // printf("%s: n_token_count_out=%zd: %s\n", __func__, n_token_count_out, LOG_TOKENS_TOSTR_PRETTY(ctx, session_tokens).c_str());

                // sanity check
                GGML_ASSERT(tokens_shared.size() <= session_tokens.size());
                for (size_t i = 0; i < tokens_shared.size(); i++) {
                    if (tokens_shared[i] != session_tokens[i]) {
                        printf("Mismatched pfx tokens [%zd]-%2X %2X %2X-%2X %2X %2X!!!!\n", i, 
                                tokens_shared[i-1], tokens_shared[i], tokens_shared[i+1],
                                session_tokens[i-1], session_tokens[i], session_tokens[i+1]);
                        return 1;
                    }
                }

                //printf("%s: token_shared=%zd - %s\n", __func__, tokens_shared.size(), LOG_TOKENS_TOSTR_PRETTY(ctx, tokens_shared).c_str());

                // remove any "future" tokens that we might have inherited from the previous session
                llama_kv_cache_seq_rm(ctx, -1, tokens_shared.size(), -1);

            } else {
                printf("%s: Load state file failed: %s\n", __func__, xbparams.pfx_file.c_str());
                session_tokens.resize(0);
                tokens_shared.clear();
                xbparams.pfx_shared = "";

                // for now this plug-in should not create cache files - comment this out for cache generation
                return 1;
            }

        } else {
            // no shared prompt detected
            tokens_shared.clear();
        }

    } else {
        // No pfc mode
        tokens_shared.clear();
    }

    if (xbparams.process_affinity) {
        // set affinity for prompt eval phase
        int64_t affinity_mask = 0;        
        // for 2-8 threads use the mask for Classic cores 
        // if possible. On systems with 16 cores (32 LP)
        // then use the cores landing in the middle (yes!)
        switch (xbparams.n_threads) {
            case 2:
                if (xbparams.is_AMD_Ryzen_HX_370) {
                    // use dense cores
                    affinity_mask = 0x0000A0ul;
                } else if (xbparams.is_AMD_Ryzen_PRO_395) {
                    // use the middle cores spannning across the CPU
                    affinity_mask = 0x00018000uL;
                }
                break;
            case 4: 
                if (xbparams.is_AMD_Ryzen_HX_370) {
                    // use dense cores
                    affinity_mask = 0x0000AAul;
                } else if (xbparams.is_AMD_Ryzen_PRO_395) {
                    // use the middle cores spannning across the CPU
                    affinity_mask = 0x000AA000uL;
                }
                break;
            case 6: 
                if (xbparams.is_AMD_Ryzen_HX_370) {
                    // use dense cores
                    affinity_mask = 0x000AAAul;
                } else if (xbparams.is_AMD_Ryzen_PRO_395) {
                    // use the middle cores spannning across the CPU
                    affinity_mask = 0x004AA400uL;
                }
                break;
            case 8: 
                if (xbparams.is_AMD_Ryzen_HX_370) {
                    // use dense cores
                    affinity_mask = 0x00AAAAul;
                } else if (xbparams.is_AMD_Ryzen_PRO_395) {
                    // use the middle cores spannning across the CPU
                    affinity_mask = 0x00AAAA00uL;
                }
                break;
            default: 
                break;
        }

        xb_set_process_affinity(affinity_mask);
        // printf("%08X: ", (uint32_t)affinity_mask);
    }

    return 0;
}

int slm_inference(std::vector<uint16_t>& line_in, bool slm_verbose = false) {
    if ((line_in.size() == 0) || (line_in[0] != '@')) {
        // no work for SLM if the user typed nothing or the first character  
        // is not an '@' character

        return 0;
    }

    xbparams.prompt.clear();
    // prepare the user prompt from the app
    for (int i = 1; i < line_in.size(); i++) {
        xbparams.prompt += (char)line_in[i];
    }
    xbparams.prompt += '\0';
    printf("%s: user prompt = [%s]\n", __func__, xbparams.prompt.c_str());
    std::string full_prompt = ::trim(xbparams.custom_template_prompt);
    size_t message_index = full_prompt.find("{message}");
    if (message_index != std::string::npos) {
        full_prompt.replace(message_index, 
            std::string("{message}").length(), 
            xbparams.prompt);
    }

    std::vector<llama_token> embd_inp;
    int n_past = 0;
    int n_kv_pfx = 0;

    if (xbparams.pfc_mode) {
        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, tokens_shared.size(), -1);
        embd_inp.insert(embd_inp.end(), tokens_shared.begin(), tokens_shared.end());
        n_past = tokens_shared.size();
        n_kv_pfx = tokens_shared.size();

        // re-apply the template since it was destroyed in pfc mode
        xbparams.prompt.append("\"\n<|end|>\n<|Assistant|>\nYou:");
    } else {
        // start from a known point
        llama_kv_cache_clear(ctx);
        n_past = 0;
        n_kv_pfx = 0;
    }

    // tokenize the remaining prompt or full prompt if pfc_mode is off
    std::vector<llama_token> tokens_input = llama_tokenize(model, xbparams.prompt, false, false);

    // append the variant part of the prompt or the full prompt for non pfc mode
    embd_inp.insert(embd_inp.end(), tokens_input.begin(), tokens_input.end());

    const int n_ctx = llama_n_ctx(ctx);
    const int n_kv_req = tokens_input.size() + (xbparams.n_len - tokens_input.size() - n_kv_pfx);

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
    if (xbparams.pfc_mode) {
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

    // printf("%s: start decoding @n_past = %d - inference size = %zd\n", __func__, n_past, embd.size());
    int64_t t_start_decoding = ggml_time_us();

    // decode the remaining prompt not covered by the shared portion
    // or the full prompt in non-pfc mode

    for (int i = 0; i < (int)embd.size(); i += xbparams.n_batch) {
        int n_eval = (int) embd.size() - i;
        if (n_eval > xbparams.n_batch) {
            n_eval = xbparams.n_batch;
        }

        if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
            printf("%s : failed to eval\n", __func__);
            return 1;
        }

        n_past += n_eval;
        // printf("%s: decoded %d tokens\n", __func__, n_eval);
    }

    int64_t t_start_generation = ggml_time_us();
    //printf("Prompt TTFT = %.2fms (size = %lld)\n", 
    //    ((t_start_generation - t_start_decoding) / 1000.0f), 
    //    embd.size());

    // compute max_len output
    int max_len = std::min(xbparams.n_len, (n_past + 128));

    std::string slm_output;
    bool valid_reply = false;
    int n_tokens_generated = 0;

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

            // if (valid_reply) {
#if 0
            // enable the following printf for streaming replies
            printf("%s", token_str.c_str());
#else
            // batched output
            slm_output += token_str;
#endif
            // }

            if (token_str.find('}') != std::string::npos) {
                // force end of output since we have a valid JSON reply
                break;
            }

            // save this new token for next evaluation
            embd[0] = new_token_id;

            n_tokens_generated += 1;
            total_tokens_generated += 1;
        }

        // bump current generated token index
        n_past += 1;

        // decode the output for the new generated token
        if (llama_decode(ctx, llama_batch_get_one(&embd[0], 1, n_past, 0))) {
            printf("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    // we have reached max_len of output, hit eog char or "}"
    if (!valid_reply) {
        // reply not correctly formatted or unhelpful
        printf("%s: ***** invalid formatted reply from model *****\n", __func__);

    } else {
        // parse the reply (json format)
        json jsonObject = json::parse(slm_output.c_str());

        // Access the values
        std::string answer = jsonObject["answer"];
        std::string justification = jsonObject["justification"];

        if (slm_verbose) {
            printf("%s: \"answer\": %s\n", __func__, answer.c_str());
            printf("%s: \"justfication\": %s\n\n", __func__, justification.c_str());
        }

        line_in.clear();
        for (char c : answer) {
            line_in.push_back((uint16_t)c);
        }
        line_in.push_back(0);
    }

    slm_output.clear();

    valid_reply = false;
    fflush(stdout);

    int64_t t_end_generation = ggml_time_us();
    double t_ms = (t_end_generation - t_start_generation) / 1000.0f;
    //printf("> token generation time = %.2fms (%d) (%.2ft/s) (%.2fms)\n", 
    //    t_ms,
    //    n_tokens_generated, 
    //    n_tokens_generated / (t_ms / 1000.0f),
    //    (t_ms / n_tokens_generated));

    t_token_generation += (t_end_generation - t_start_generation);
    return 0;
}

void slm_terminate() {
#if 0

    printf("\n");

    printf("%s: generated %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, 
            total_tokens_generated, (t_token_generation / 1000000.0f), 
            total_tokens_generated / (t_token_generation / 1000000.0f));

    llama_print_timings(ctx);

#endif

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();
}

