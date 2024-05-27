#include <vector>
#include <iostream>
#include <chrono>

#include "common.h"
#include "llama.h"

using namespace std;

int main() {
    auto seed = 42;
    auto thread_count = 12;
    auto last_n_tokens_size = 64;
    auto prompt = "The quick brown fox";
    auto model_path = "./models/Phi-3-mini-4k-instruct-Q2_K.gguf";

    auto n_past = 0;
    auto last_n_tokens_data = vector<llama_token>(last_n_tokens_size, 0);

    // init
    auto params = llama_context_default_params();
    params.seed = seed;
    auto ctx = llama_init_from_file(model_path, params);
    auto tokens = vector<llama_token>(params.n_ctx);
    auto n_prompt_tokens = llama_tokenize(ctx, prompt, tokens.data(), tokens.size(), true);

    if (n_prompt_tokens < 1) {
        cout << "Failed to tokenize prompt" << endl;
        return 1;
    }

    // evaluate prompt

    // llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past, thread_count);

    last_n_tokens_data.insert(last_n_tokens_data.end(), tokens.data(), tokens.data() + n_prompt_tokens);
    n_past += n_prompt_tokens;

    // save kv state, last n tokens and n_past
    auto kv_cache_size = llama_get_kv_cache_size(ctx);
    auto kv_cache_token_count = llama_get_kv_cache_token_count(ctx);
    auto kv_cache = llama_get_kv_cache(ctx);
    auto kv_cache_saved = vector<uint8_t>(kv_cache, kv_cache + kv_cache_size);

    auto last_n_tokens_data_saved = vector<llama_token>(last_n_tokens_data);
    auto n_past_saved = n_past;

    // save first generated token
    auto first_generated_token = llama_token(0);

    // first run
    cout << endl << prompt;
    for (auto i = 0; i < 6; i++) {
        auto next_token = llama_sample_top_p(
            ctx,
            &last_n_tokens_data.back() - last_n_tokens_size,
            last_n_tokens_size,
            40,
            1.0,
            1.0,
            1.1
        );
        if (i == 0) {
            first_generated_token = next_token;
        }
        auto next_token_str = llama_token_to_str(ctx, next_token);
        last_n_tokens_data.push_back(next_token);
        cout << next_token_str;
        if (llama_eval(ctx, &next_token, 1, n_past, thread_count)) {
            cout << endl << "Failed to evaluate" << endl;
            return 1;
        }
        n_past += 1;
    }
    cout << endl << endl;

    // free old model
    llama_free(ctx);

    // load new model
    params = llama_context_default_params();
    params.seed = seed;
    ctx = llama_init_from_file(model_path, params);

    // restore state
    llama_set_kv_cache(ctx, kv_cache_saved.data(), kv_cache_size, kv_cache_token_count);
    last_n_tokens_data = last_n_tokens_data_saved;
    n_past = n_past_saved;

    // restore first generated token so we can safely sample
    // llama_eval(ctx, &first_generated_token, 1, n_past, thread_count);
    last_n_tokens_data.push_back(first_generated_token);
    n_past += 1;

    // second run
    cout << endl << prompt << llama_token_to_str(ctx, first_generated_token);
    for (auto i = 0; i < 5; i++) {
        auto next_token = llama_sample_top_p(
            ctx,
            &last_n_tokens_data.back() - last_n_tokens_size,
            last_n_tokens_size,
            40,
            1.0,
            1.0,
            1.1
        );
        auto next_token_str = llama_token_to_str(ctx, next_token);
        last_n_tokens_data.push_back(next_token);
        cout << next_token_str;
        //if (llama_eval(ctx, &next_token, 1, n_past, thread_count)) {
        //    cout << endl << "Failed to evaluate" << endl;
        //    return 1;
        //}
        n_past += 1;
    }
    cout << endl << endl;
    return 0;
}
