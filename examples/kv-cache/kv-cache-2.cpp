#include <vector>
#include <iostream>
#include <chrono>

#include "common.h"
#include "llama.h"

//#include "llama.cpp"

using namespace std;

int main(int argc, char ** argv) {
    auto seed = 42;
    auto thread_count = 4;
    auto last_n_tokens_size = 64;
    auto prompt = "The quick brown fox";
    auto model_path = "models/Phi-3-mini-4k-instruct-Q2_K.gguf";


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

    llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past, thread_count);

    last_n_tokens_data.insert(last_n_tokens_data.end(), tokens.data(), tokens.data() + n_prompt_tokens);
    n_past += n_prompt_tokens;

    // Save state (rng, logits, embedding and kv_cache) to file
    FILE *fp_write = fopen("dump_state.bin", "wb");
    auto state_size = llama_get_state_size(ctx);
    auto state_mem = new uint8_t[state_size];
    llama_copy_state_data(ctx, state_mem); // could also copy directly to memory mapped file
    fwrite(state_mem, 1, state_size, fp_write);
    fclose(fp_write);

    // save state (last tokens)
    auto last_n_tokens_data_saved = vector<llama_token>(last_n_tokens_data);
    auto n_past_saved = n_past;

    // save first generated token
    auto first_generated_token = llama_token(0);

    // first run
    cout << endl
         << prompt;
    for (auto i = 0; i < 6; i++) {
        auto next_token = llama_sample_top_p_top_k(
            ctx,
            &last_n_tokens_data.back() - last_n_tokens_size,
            last_n_tokens_size,
            40,
            1.0,
            1.0,
            1.1);
        if (i == 0) {
            first_generated_token = next_token;
        }
        auto next_token_str = llama_token_to_str(ctx, next_token);
        last_n_tokens_data.push_back(next_token);
        cout << next_token_str;
        if (llama_eval(ctx, &next_token, 1, n_past, thread_count)) {
            cout << endl
                 << "Failed to evaluate" << endl;
            return 1;
        }
        n_past += 1;
    }
    cout << endl
         << endl;

    // free old model
    llama_free(ctx);

    // load new model
    params = llama_context_default_params();
    params.seed = seed;

    auto ctx2 = llama_init_from_file(model_path, params);

    // Load state (rng, logits, embedding and kv_cache) from file
    FILE *fp_read = fopen("dump_state.bin", "rb");
    auto state_size2 = llama_get_state_size(ctx2);
    if (state_size != state_size2) {
        cerr << "state size differs\n";
    }
    fread(state_mem, 1, state_size, fp_read);
    llama_set_state_data(ctx2, state_mem);  // could also read directly from memory mapped file
    fclose(fp_read);

    // restore state (last tokens)
    last_n_tokens_data = last_n_tokens_data_saved;
    n_past = n_past_saved;

    // this should not be necessary with llama_copy_state_data & llama_set_state_data as they will save and restore logits.
    
    // // restore first generated token so we can safely sample
    // llama_eval(
    //     ctx2,
    //     &first_generated_token,
    //     1,
    //     n_past,
    //     thread_count);
    // last_n_tokens_data.push_back(first_generated_token);
    // n_past += 1;
    // cout << endl << prompt << llama_token_to_str(ctx2, first_generated_token);
    
    // second run
    for (auto i = 0; i < 5; i++) {
        auto next_token = llama_sample_top_p_top_k(
            ctx2,
            &last_n_tokens_data.back() - last_n_tokens_size,
            last_n_tokens_size,
            40,
            1.0,
            1.0,
            1.1);
        auto next_token_str = llama_token_to_str(ctx2, next_token);
        last_n_tokens_data.push_back(next_token);
        cout << next_token_str;
        if (llama_eval(ctx2, &next_token, 1, n_past, thread_count)) {
            cout << endl
                 << "Failed to evaluate" << endl;
            return 1;
        }
        n_past += 1;
    }
    cout << endl
         << endl;
    return 0;
}
