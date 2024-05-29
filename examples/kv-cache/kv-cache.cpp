#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <thread>

struct gpt_params {
    uint32_t seed                    = 42; // RNG seed - default was 0xFFFFFFFF
    int32_t n_threads                = 4;
    int32_t n_threads_batch          = -1;    // number of threads to use for batch processing (-1 = use n_threads)
    std::string model                = "";  // model path
    std::string prompt               = "";
    std::string prompt_file          = "";  // store the external prompt file name
    std::string path_prompt_cache    = "";  // path to file for saving/loading prompt eval state
    std::string script               = "";  // copied script input file
    std::string custom_p_file        = "";  // custom prompts input file
    std::string pfx_cache_dir        = "shared_prefix"; // default prefix cache directory
    std::string pfx_cache_file       = "";  // default prefix cache file 
};

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
        GGML_ASSERT(check == -n_tokens);
    } else {
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

void llama_batch_add(
                 struct llama_batch & batch,
                        llama_token   id,
                          llama_pos   pos,
    const std::vector<llama_seq_id> & seq_ids,
                               bool   logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;

    batch.n_tokens++;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT]\n" , argv[0]);
        return 1 ;
    }

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (argc >= 3) {
        params.prompt = argv[2];
    }

    if (params.prompt.empty()) {
        params.prompt = "The capital of Portugal is";
    }

    unsigned int n_threads = std::thread::hardware_concurrency();
    if (n_threads > 0) {
        params.n_threads =(n_threads <= 4) ? n_threads : (n_threads / 2);
    }

    // total length of the sequence including the prompt
    const int n_len = 32;

    // init LLM

    llama_backend_init();
    //- llama_numa_init(params.numa);

    // initialize the model

    llama_model_params model_params = llama_model_default_params();

    // model_params.n_gpu_layers = 99; // offload all layers to the GPU

    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        printf("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed  = 42;
    ctx_params.n_ctx = 1024;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    llama_context * ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        printf("%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // tokenize the prompt

    std::vector<llama_token> tokens_list;
    tokens_list = llama_tokenize(llama_get_model(ctx), params.prompt, true);

    const int n_ctx    = llama_n_ctx(ctx);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    printf("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);
    printf("%s: n_threads = %d, n_threads_batch = %d\n", __func__, ctx_params.n_threads, ctx_params.n_threads_batch);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        printf("%s: error: n_kv_req > n_ctx, the required KV cache size is not big enough\n", __func__);
        printf("%s:        either reduce n_len or increase n_ctx\n", __func__);
        return 1;
    }

    // print the prompt token-by-token

    printf("\n");

    for (auto id : tokens_list) {
        printf("%s", llama_token_to_piece(ctx, id).c_str());
    }

    fflush(stdout);

    // create a llama_batch with size 512
    // we use this object to submit token data for decoding

    llama_batch batch = llama_batch_init(512, 0, 1);

    // evaluate the initial prompt
    for (size_t i = 0; i < tokens_list.size(); i++) {
        llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        printf("%s: llama_decode() failed\n", __func__);
        return 1;
    }

    // main loop

    int n_cur    = batch.n_tokens;
    int n_decode = 0;

    const auto t_main_start = ggml_time_us();

    while (n_cur <= n_len) {
        // sample the next token
        {
            auto   n_vocab = llama_n_vocab(model);
            auto * logits  = llama_get_logits_ith(ctx, batch.n_tokens - 1);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
            }

            llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

            // sample the most likely token (greedy sampling algo)
            const int   top_k = 40;
            const float top_p = 0.9f;
            const float temp  = 0.1f;

            llama_sample_top_k(ctx, &candidates_p, top_k, 1);
            llama_sample_top_p(ctx, &candidates_p, top_p, 1);
            llama_sample_temp (ctx, &candidates_p, temp);

            const llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation - are we done?
            if (llama_token_is_eog(model, new_token_id) || n_cur == n_len) {
                printf("\n");
                break;
            }

            printf("%s", llama_token_to_piece(ctx, new_token_id).c_str());
            fflush(stdout);

            // prepare the next batch
            batch.n_tokens = 0;

            // push this new token for next evaluation
            llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

            n_decode += 1;
        }

        n_cur += 1;

        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            printf("%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    printf("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    llama_print_timings(ctx);
    print_tensor_op_perf_data();
    printf("\n");

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
