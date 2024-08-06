#include <vector>
#include <iostream>
#include <chrono>

// #include "common.h"
#include "llama.h"

using namespace std;

std::vector<llama_token> llama_tokenize(
    const struct llama_model * model,
    const std::string & text,
    bool add_special,
    bool parse_special) {
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

int main()
{
    auto seed = 42;
    auto last_n_tokens_size = 1532;
    std::string prompt =
"Context:"
"You are a helpful assistant for Mara, a 20 - year old adolescent female."
"Mara is not comfortable answering questions so be patient with her answers."
"Mara is currently standing alone in front of a closed door of a room on the upper floor of her childhood home."
"Mara has a sister named Izzy. The closed door Mara is facing is Izzy's bedroom, not Mara's bedroom."
"Mara's bedroom is located downstairs on the ground floor, not on the upper floor."
" "
"Task:"
"You started the conversation with this initial question\"Mara, what's behind this closed door?\""
"Your task is to respond to Mara in a helpful manner to get Mara to talk about her sister and answer questions from you."
"Before you respond to Mara, you'll be given a list of options to choose from."
"You must choose one of the options to respond to Mara in a way that does NOT directly force her to talk about her sister."
"If Mara makes up incorrect answer, pick the best option to correct her but DO NOT reply with the correct answer."
"If Mara answers the question correctly, you MUST pick the best option to AFFIRM her answer."
"Generate ONLY ONE answer and format it in ONE SINGLE JSON format with these two fields:"
"* answer: the option you chose to respond to Mara"
"* justification: a brief explanation of why you chose the option in this scenario"
" "
"Note: Only generate one json response each time and then STOP right after the json response."
" "
"<|end|>"
"Conversation:"
"<|user|>"
"Options: ["
"\"a. Are you sure Mara, what makes you believe that is true?\","
"\"b. You might be confused Mara, your room is on the ground floor.\","
"\"c. That is correct Mara. It is your sister's room\","
"\"d. Sorry Mara, that is not precise enough. Can you please elaborate.\","
"\"e. Are you trying to avoid the question?\""
"]"
"Mara: \"I am not sure I know\""
"<|end|>"
"<|assistant|>"
"You: {"
"    \"answer\": \"d. Sorry Mara, that is not precise enough. Can you please elaborate.\","
"    \"justification\": \"Mara is not willingly to answer it's her sister Izzy room. Option (d) invites Mara to be more open.\""
"}"
"<|end|>"
"<|user|>"
"Mara: \"This is my dad's room.\""
"<|end|>"
"<|assistant|>"
"You:";

    auto model_path = "./models/Phi-3-mini-4k-instruct-Q2_K.gguf";

    auto n_past = 0;
    auto last_n_tokens_data = vector<llama_token>(last_n_tokens_size, 0);
    auto last_n_tokens_data_saved = vector<llama_token>(last_n_tokens_data);
    auto tokens = vector<llama_token>(0);
    auto model_params = llama_model_default_params();
    auto cparams = llama_context_default_params();
    cparams.seed = seed;
    cparams.n_threads = 16;
    cparams.n_ctx = 2048;

    FILE *fp = fopen("dump_state.bin", "rb");
    if (fp != NULL) {
        fclose(fp);
        printf("Skipped to 2nd round\n");
        goto second_round;
    }

    // init
   // initialize the model
    auto model = llama_load_model_from_file(model_path, model_params);
    tokens = vector<llama_token>(cparams.n_ctx);
    tokens = llama_tokenize(model, prompt, false, false);

    auto ctx = llama_new_context_with_model(model, cparams);
    auto n_prompt_tokens = tokens.size();
    
    if (n_prompt_tokens < 1) {
        cout << "Failed to tokenize prompt" << endl;
        return 1;
    }

    // evaluate prompt
    printf("%s: Starting 1st decode...\n", __func__);
    // llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past);
    for (int i = 0; i < (int)tokens.size(); i += cparams.n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > (int)cparams.n_batch) {
            n_eval = cparams.n_batch;
        }

        if (llama_decode(ctx, llama_batch_get_one(&tokens[i], n_eval, n_past, 0))) {
            printf("%s : failed to eval\n", __func__);
            return 1;
        }

        n_past += n_eval;
        printf("%s: n_past = %d - n_eval = %d\n", __func__, n_past, n_eval);
    }

    last_n_tokens_data.insert(last_n_tokens_data.end(), tokens.data(), tokens.data() + n_prompt_tokens);
    n_past += n_prompt_tokens;

    printf("%s: saving state...\n", __func__);
    // Save state (rng, logits, embedding and kv_cache) to file
    FILE *fp_write = fopen("dump_state.bin", "wb");
    auto state_size = llama_state_get_size(ctx);
    auto state_mem = new uint8_t[state_size];
    llama_state_get_data(ctx, state_mem);
    fwrite(state_mem, 1, state_size, fp_write);
    fclose(fp_write);
    printf("%s: DONE saving state...\n", __func__);

    // save state (last tokens)
    auto n_past_saved = n_past;

    // save first generated token
    // auto first_generated_token = llama_token(0);

    // first run
    cout << endl
         << prompt;

    printf("\nStart first generation phase...\n");

    for (auto i = 0; i < 128; i++) {
        {
            auto n_vocab = llama_n_vocab(model);
            auto* logits = llama_get_logits_ith(ctx, 0);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);

            for (llama_token token_id = 0; token_id < n_vocab; token_id++)
            {
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

            llama_token new_token_id = llama_sample_token_greedy(ctx, &candidates_p);

            // is it an end of generation - are we done?
            if (llama_token_is_eog(model, new_token_id)) {
                printf("\n");
                break;
            }

            const std::string token_str = llama_token_to_piece(ctx, new_token_id);
            printf("%s", token_str.c_str());
            fflush(stdout);

            last_n_tokens_data.push_back(new_token_id);

            // if (llama_eval(ctx, &new_token_id, 1, n_past))
            if (llama_decode(ctx, llama_batch_get_one(&new_token_id, 1, n_past, 0))) {
                printf("%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }
        }
        n_past += 1;
    }
    cout << endl
         << endl;

    printf("DONE - first generation phase...\n");
    
    // free old model
    llama_free(ctx);

second_round:
    // load new model
    auto cparams2 = llama_context_default_params();
    cparams2.seed = seed;
    cparams2.n_threads = 16;
    cparams2.n_ctx = 2048;
    auto model2 = llama_load_model_from_file(model_path, model_params);
    auto ctx2 = llama_new_context_with_model(model2, cparams2);

    std::string prompt2 =
"<|user|>"
"Mara: \"I am not sure I know\""
"<|end|>"
"<|assistant|>"
"You: {"
"    \"answer\": \"d. Sorry Mara, that is not precise enough. Can you please elaborate.\","
"    \"justification\": \"Mara is not willingly to answer it's her sister Izzy room. Option (d) invites Mara to be more open.\""
"}"
"<|end|>"
"<|user|>"
"Mara: \"This is my brother's room.\""
"<|end|>"
"<|assistant|>"
"You:";

    printf("%s: start Loading dump file...\n", __func__);
    // Load state (rng, logits, embedding and kv_cache) from file
    FILE *fp_read = fopen("dump_state.bin", "rb");
    auto state_size2 = llama_state_get_size(ctx2);
    auto state_mem2 = new uint8_t[state_size2];
    //if (state_size != state_size2) {
    //    cerr << "state size differs\n";
    //}
    fread(state_mem2, 1, state_size2, fp_read);
    llama_state_set_data(ctx2, state_mem2);  // could also read directly from memory mapped file
    fclose(fp_read);
    printf("%s: DONE Loading dump file...\n", __func__);

    auto tokens2 = vector<llama_token>(cparams2.n_ctx);
    tokens2 = llama_tokenize(model2, prompt2, false, false);

    auto n_prompt_tokens2 = tokens2.size();
    n_past = 0;

    if (n_prompt_tokens2 < 1) {
        cout << "Failed to tokenize prompt2" << endl;
        return 1;
    }

    // evaluate prompt
    printf("%s: Starting 2nd decode...\n", __func__);
    // llama_eval(ctx, tokens.data(), n_prompt_tokens, n_past);
    for (int i = 0; i < (int)tokens2.size(); i += cparams2.n_batch) {
        int n_eval = (int) tokens2.size() - i;
        if (n_eval > (int)cparams2.n_batch) {
            n_eval = cparams2.n_batch;
        }

        if (llama_decode(ctx2, llama_batch_get_one(&tokens2[i], n_eval, n_past, 0))) {
            printf("%s : failed to eval\n", __func__);
            return 1;
        }

        n_past += n_eval;
        printf("%s: n_past = %d - n_eval = %d\n", __func__, n_past, n_eval);
    }

    last_n_tokens_data.insert(last_n_tokens_data.end(), tokens2.data(), tokens2.data() + n_prompt_tokens2);
    n_past += n_prompt_tokens2;

    // restore state (last tokens)
    // last_n_tokens_data = last_n_tokens_data_saved;
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

    printf("%s: start 2nd txt generation phase...\n", __func__);
    //llama_kv_cache_clear(ctx2);

    // second run
    for (auto i = 0; i <128; i++) {
        {
            auto n_vocab = llama_n_vocab(model2);
            auto* logits = llama_get_logits_ith(ctx2, 0);

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

            llama_sample_top_k(ctx2, &candidates_p, top_k, 1);
            llama_sample_top_p(ctx2, &candidates_p, top_p, 1);
            llama_sample_temp(ctx2, &candidates_p, temp);

            llama_token new_token_id = llama_sample_token_greedy(ctx2, &candidates_p);

            // is it an end of generation - are we done?
            if (llama_token_is_eog(model2, new_token_id)) {
                printf("\n");
                break;
            }

            const std::string token_str = llama_token_to_piece(ctx2, new_token_id);
            printf("%s", token_str.c_str());
            fflush(stdout);

            last_n_tokens_data.push_back(new_token_id);

            //if (llama_eval(ctx2, &new_token_id, 1, n_past))
            if (llama_decode(ctx2, llama_batch_get_one(&new_token_id, 1, n_past, 0))) {
                printf("%s : failed to eval, return code %d\n", __func__, 1);
                return 1;
            }
        }
        n_past += 1;
    }
    cout << endl
        << endl;
    printf("%s: DONE 2nd txt generation phase...\n", __func__);

    llama_free(ctx2);
    return 0;
}
