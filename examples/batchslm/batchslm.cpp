// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <unistd.h>
#elif defined (_WIN32)
#if !defined WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

std::string custom_template_prompt;
std::vector<std::string> custom_prompts;

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    if (str.empty()) {
        return str; // Return early if the string is empty
    }

    size_t start = 0;
    size_t end = str.length();

    // trim leading white space
    while ((start < end) && isspace(str[start])) {
        start += 1;
    }

    // trim trailing white space
    while ((end > start) && isspace(str[end - 1])) {
        end -= 1;
    }

    GGML_ASSERT(end >= start);
    return str.substr(start, end - start);
}

static std::string k_system =
R"(Transcript of a never ending dialog, where the User interacts with an Assistant.
The Assistant is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.
Your answer should never be more than 128 characters.

User: Who is The Little Prince?
Assistant: The Little Prince is they key figure in a novella written and illustrated by Saint-Exupery published in 1943. The story offers profound observations about life and human nature, making it a beloved classic for readers of all ages. 
User: What is B612?
Assistant: B612 is the name of the asteroid where the Little Prince lives. B612 is a central part of the story, symbolizing the Little Prince's home and his love for his rose. It's a place of simplicity and beauty, reflecting the themes of the book.".
User:)";

static std::vector<std::string> k_prompts = {
    "What is the meaning of life?",
    "Tell me an interesting fact about The Little Prince character.",
    "How can one travel to a different dimension to meet your other self?",
    "Are you familiar with the Beegees and can you tell me more about them?",
    "Can you recommend some books with the same style as Silk the novella by Alessandro Baricco?",
    "What is the best way to bake a bread pudding?",
    "How to get a job that allows you to look at people's eyes all day long?",
    "If you could be a animal, what would it be?",
    "I want to watch a movie that helps me find peace about the world.",
    "Who wrote the song Pale Blue Eyes?",
    "What is the most important ingredient to make a pizza?",
};

struct client {
    ~client() {
        if (ctx_sampling) {
            llama_sampling_free(ctx_sampling);
        }
    }

    int32_t id = 0;

    llama_seq_id seq_id = -1;

    llama_token sampled;

    int64_t t_start_prompt;
    int64_t t_start_gen;
    bool valid_reply = false;

    int32_t n_prompt  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;

    std::string input;
    std::string prompt;
    std::string response;

    struct llama_sampling_context * ctx_sampling = nullptr;
};

// Define a split string function to ...
static std::vector<std::string> split_string(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

bool processCustomPromptsFromFile(const std::string& custom_p_file) {
    std::ifstream cpfile(custom_p_file);
    if (!cpfile.is_open()) {
        printf("[%s]: failed to open [%s]\n", __func__, custom_p_file.c_str());
        return false;
    }

    std::string line;

    bool templatePromptMode = false;
    bool userPromptMode = false;
    custom_prompts.clear();
    custom_template_prompt = "";

    // process CUSTOM_SYSTEM_PROMPT
    while (std::getline(cpfile, line)) {
        if (line == "CUSTOM_TEMPLATE_PROMPT") {
            templatePromptMode = true;
            continue;
        } else if (line == "CUSTOM_PROMPT") {
            userPromptMode = true;
            continue;
        } else if (line == "END_SECTION") {
            templatePromptMode = false;
            userPromptMode = false;
            continue;
        }

        if (templatePromptMode) {
            custom_template_prompt += line + '\n';
        } else if (userPromptMode) {
            line.erase(std::remove(line.begin(), line.end(), '\"'), line.end());
            custom_prompts.push_back(::trim(line));
        }
    }

    cpfile.close();

    return true;
}

std::string pfx_file_path(std::string pfx, std::string dir, std::string file) {
    static std::hash<std::string> hasher;
    std::string generated_name = std::to_string(hasher(pfx));

    // create the cache dir if it does not exist yet
    if (!CreateDirectoryA(dir.c_str(), NULL)) {
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            fprintf(stderr, "%s: Failed to create directory: %s - use current dir for prefix cache\n",
                __func__, dir.c_str());
            dir = ".";
        }
    }

    // default generated file name
    std::string full_file_path = dir + "/" + generated_name;

    if (file != "default") {
        full_file_path = dir + "/" + file;
    }

    return full_file_path;
}

bool pfc_init(gpt_params& params, 
              llama_context * ctx,
              std::string shared_context_string, 
              std::vector<llama_token> tokens_shared) {
    bool rc = true;

    const llama_model * model = llama_get_model(ctx);
    std::string pfx_cache_dir = "./llama_cache";

    // start from a known point
    llama_kv_cache_clear(ctx);

    // build the shared prompt
    // tokenize(a) + tokenize(b) != tokenize(a+b), we tokenize pfx and content separately
    // std::vector<llama_token> tokens_pfx = llama_tokenize(model, shared_context_string, false, false);

    // build the cache file directory
    std::string pfx_file = pfx_file_path(shared_context_string, pfx_cache_dir, params.pfx_cache_file);

    // load the cache and create one if it does not exist
    std::vector<llama_token> session_tokens;
    session_tokens.resize(params.n_ctx);
    // indicate first time through to reduce output
    size_t n_token_count_out = 0xffffffff;
    if (llama_state_load_file(ctx, 
                              pfx_file.c_str(),
                              session_tokens.data(),
                              session_tokens.capacity(),
                              &n_token_count_out)) {
        printf("%s: Loading saved state successfully from '%s' (size %zd)...\n", __func__, pfx_file.c_str(), tokens_shared.size());
        session_tokens.resize(n_token_count_out);
        llama_set_rng_seed(ctx, params.seed);
        // printf("%s: n_token_count_out=%zd: %s\n", __func__, n_token_count_out, LOG_TOKENS_TOSTR_PRETTY(ctx, session_tokens).c_str());

        // sanity check
        GGML_ASSERT(tokens_shared.size() <= session_tokens.size());
        for (size_t i = 0; i < tokens_shared.size(); i++) {
            if (tokens_shared[i] != session_tokens[i]) {
                printf("Mismatched pfx tokens [%zd]-(%2X %2X %2X)-(%2X %2X %2X)!\n", i, 
                    tokens_shared[i-1], tokens_shared[i], tokens_shared[i+1],
                    session_tokens[i-1], session_tokens[i], session_tokens[i+1]);
                rc = false;
            }
        }

        //printf("%s: token_shared=%zd - %s\n", __func__, tokens_shared.size(), LOG_TOKENS_TOSTR_PRETTY(ctx, tokens_shared).c_str());

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, tokens_shared.size(), -1);

    } else {
        printf("%s: Load state file failed: %s\n", __func__, pfx_file.c_str());
        rc = false;
    }

    return rc;
}

int main(int argc, char ** argv) {
    gpt_params params;

    ggml_time_init();
    const auto t_main_start = ggml_time_us();

    srand(params.seed);

    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    // number of simultaneous "clients" to simulate
    const int32_t n_clients = params.n_parallel;

    // dedicate one client as the system prompt
    params.n_parallel += 1;

    // requests to simulate
    const int32_t n_seq = params.n_sequences;

    // insert new requests as soon as the previous one is done
    const bool cont_batching = params.cont_batching;

    const bool dump_kv_cache = params.dump_kv_cache;

    // init llama.cpp
    llama_backend_init();

    llama_model * model = NULL;
    llama_context * ctx = NULL;

#if 0

    // load the target model
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    if (ctx == NULL) {
        printf("%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

// 0000004f`da67e600 00007ff6`b4f985ca     batchslmza!ggml_compute_forward_mul_mat+0x1e2 [xbox\gamecore\so2001\z-slmapp\za-ggml\ggml.c @ 13450]
// 0000004f`da67e7a0 00007ff6`b4f98f31     batchslmza!ggml_graph_compute_thread+0x19a [xbox\gamecore\so2001\z-slmapp\za-ggml\ggml.c @ 20245]
// 0000004f`da67e810 00007ff6`b4f9c544     batchslmza!ggml_graph_compute+0x351 [xbox\gamecore\so2001\z-slmapp\za-ggml\ggml.c @ 20524]
// 0000004f`da67e9e0 00007ff6`b4f9ea1c     batchslmza!ggml_backend_cpu_graph_compute+0xa4 [xbox\gamecore\so2001\z-slmapp\za-ggml\ggml-backend.c @ 812]
// 0000004f`da67ea60 00007ff6`b4f3f2f3     batchslmza!ggml_backend_sched_compute_splits+0x14c [xbox\gamecore\so2001\z-slmapp\za-ggml\ggml-backend.c @ 1777]
// 0000004f`da67eb70 00007ff6`b4f4d2ff     batchslmza!llama_decode_internal+0xba3 [xbox\gamecore\so2001\z-slmapp\batchslm\llama.cpp @ 12585]
// 0000004f`da67ed70 00007ff6`b4ec0c9a     batchslmza!llama_decode+0x2f [xbox\gamecore\so2001\z-slmapp\batchslm\llama.cpp @ 18376]
// 0000004f`da67edf0 00007ff6`b4ea51f1     batchslmza!llama_init_from_gpt_params+0x93a [xbox\gamecore\so2001\z-slmapp\batchslm\common.cpp @ 2476]
// 0000004f`da67efa0 00007ff6`b4fcd263     batchslmza!main+0xf1 [xbox\gamecore\so2001\z-slmapp\batchslm\batchslm.cpp @ 283]
// (Inline Function) --------`--------     batchslmza!invoke_main+0x22 [VCCRT\vcstartup\src\startup\exe_common.inl @ 78]
// 0000004f`da67fda0 00007ff8`0381257d     batchslmza!__scrt_common_main_seh+0x10b [VCCRT\vcstartup\src\startup\exe_common.inl @ 288]
// 0000004f`da67fde0 00007ff8`0462af28     KERNEL32!BaseThreadInitThunk+0x1d
// 0000004f`da67fe10 00000000`00000000     ntdll!RtlUserThreadStart+0x28

#else

    // initialize the model with more customized params
    llama_model_params model_params = llama_model_default_params();

    model = llama_load_model_from_file(params.model.c_str(), model_params);
    if (model == NULL) {
        printf("%s: error: unable to load model\n" , __func__);
        return 1;
    }

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();

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

#endif

    // load the prompts from an external file if there are any
    if (params.custom_prompts_on) {
        // cpf mode
        printf("[%s]: processing cpf input file [%s]\n", __func__, params.custom_p_file.c_str());
        processCustomPromptsFromFile(params.custom_p_file);
        // reset system prompt to using the custom template prompt instead
        k_system = custom_template_prompt;
        size_t pos = custom_template_prompt.find("{message}");
        if (pos != std::string::npos) {
            // build the shared context for all prompts
            k_system = ::trim(custom_template_prompt.substr(0, pos));
        }
        // refresh prompts with custom_prompts
        int index = 0;
        for (const auto& prompt : custom_prompts) {
            k_prompts.resize(index + 1);
            k_prompts[index] = prompt;
            index++;
        }
        printf("    cpf custom prompts %3d\n", index);
        printf("    cpf system prompt size %3I64d \n", k_system.length());

    } else if (params.prompt.empty()) {
        printf("\nNo external prompt file so proceed with build-in defaults...\n");

    } else {
        // Output each line of the input params.prompts vector and copy to k_prompts
        int index = 0;
        printf("\nProcessing the external prompt file %s\n\n", params.prompt_file.c_str());
        std::vector<std::string> prompts = split_string(params.prompt, '\n');
        for (const auto& prompt : prompts) {
            k_prompts.resize(index + 1);
            k_prompts[index] = prompt;
            index++;
        }
        printf("    %3d external prompts ingested\n", index);
    }

    printf("\n\n");

    const int n_ctx = llama_n_ctx(ctx);

    std::vector<client> clients(n_clients);
    for (size_t i = 0; i < clients.size(); ++i) {
        auto & client = clients[i];
        client.id = i;
        client.ctx_sampling = llama_sampling_init(params.sparams);
    }

    std::vector<llama_token> tokens_system;
    // tokens_system = ::llama_tokenize(ctx, k_system, true);
    tokens_system = llama_tokenize(model, k_system, false, false);
    const int32_t n_tokens_system = tokens_system.size();

    llama_seq_id g_seq_id = 0;

    // the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
    // users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    int32_t n_total_prompt = 0;
    int32_t n_total_gen    = 0;
    int32_t n_cache_miss   = 0;

    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, n_clients);

    if (params.use_omp) {
        ggml_select_omp();
    }

    // LOG_TEE("%s: Simulating parallel requests from clients:\n", __func__);
    LOG_TEE("%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d, context = %d\n\n", 
        __func__, n_clients, n_seq, cont_batching, n_tokens_system, n_ctx);

    if (params.use_prefix_cache && 
        pfc_init(params, ctx, k_system, tokens_system)) {
        // if prefix cache is specified then initialize it
        for (int32_t i = 0; i < n_tokens_system; ++i) {
            llama_batch_add(batch, tokens_system[i], i, { 0 }, false);
        }

        // assign the system KV cache to all parallel sequences
        for (int32_t i = 1; i <= n_clients; ++i) {
            llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
        }            

    } else {
        LOG_TEE("%s: Evaluating the system prompt ...\n", __func__);
        const auto t_eval_start = ggml_time_us();

        for (int32_t i = 0; i < n_tokens_system; ++i) {
            llama_batch_add(batch, tokens_system[i], i, { 0 }, false);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        // assign the system KV cache to all parallel sequences
        for (int32_t i = 1; i <= n_clients; ++i) {
            llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
        }
        const auto t_eval_end = ggml_time_us();
        LOG_TEE("%s: system prompt evaluation DONE in %5.2fs\n", 
            __func__,
            (t_eval_end - t_eval_start) / 1e6);
    }

    LOG_TEE("%s: Processing parallel requests ...\n\n", __func__);

    while (true) {
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            llama_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        // prepare the batch of requests
        llama_batch_clear(batch);

        // decode any currently ongoing sequences
        for (auto & client : clients) {
            if (client.seq_id == -1) {
                // this client is done
                continue;
            }

            client.i_batch = batch.n_tokens;

            llama_batch_add(batch, client.sampled, n_tokens_system + client.n_prompt + client.n_decoded, { client.id + 1 }, true);

            client.n_decoded += 1;
        }

        if (batch.n_tokens == 0) {
            // all sequences have ended - clear the entire KV cache
            for (int i = 1; i <= n_clients; ++i) {
                llama_kv_cache_seq_rm(ctx, i, -1, -1);
                // but keep the system prompt
                llama_kv_cache_seq_cp(ctx, 0, i, -1, -1);
            }

            // LOG_TEE("%s: clearing the KV cache\n", __func__);
        }

        // insert new sequences for decoding
        if (cont_batching || batch.n_tokens == 0) {
            for (auto & client : clients) {
                if ((client.seq_id == -1) && (g_seq_id < n_seq)) {
                    client.seq_id = g_seq_id;

                    client.t_start_prompt = ggml_time_us();
                    client.t_start_gen    = 0;

                    client.input    = k_prompts[g_seq_id % k_prompts.size()];
                    if (params.custom_prompts_on) {
                        client.prompt   = client.input + "\"\n<|end|>\n<|Assistant|>\nYou:";
                    } else {
                        client.prompt   = client.input + "\nAssistant:";
                    }
                    client.response = "";

                    llama_sampling_reset(client.ctx_sampling);

                    // no need to prepend BOS because we have a system prompt!
                    std::vector<llama_token> tokens_prompt;
                    tokens_prompt = ::llama_tokenize(ctx, client.prompt, false);

                    for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                        llama_batch_add(batch, tokens_prompt[i], i + n_tokens_system, { client.id + 1 }, false);
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    client.n_prompt  = tokens_prompt.size();
                    client.n_decoded = 0;
                    client.valid_reply = false;
                    client.i_batch   = batch.n_tokens - 1;

                    // LOG_TEE("Client %3d, seq %4d, started decoding ...\n", client.id, client.seq_id);

                    g_seq_id += 1;

                    // insert new requests one-by-one
                    // if (cont_batching) {
                    //     break;
                    //}
                }
            }
        }

        // printf(".");

        if (batch.n_tokens == 0) {
            break;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            // experiment: process in powers of 2
            // if (i + n_batch > (int32_t) batch.n_tokens && n_batch > 32) {
            //     n_batch /= 2;
            //     i -= n_batch;
            //     continue;
            //}

            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_TEE("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return 1;
                }

                LOG("%s : failed to decode the batch, retrying with n_batch = %d\n", __func__, n_batch / 2);

                n_cache_miss += 1;

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                continue;
            }

            // LOG_TEE("%s : decoded batch of %d tokens\n", __func__, n_tokens);

            for (auto & client : clients) {
                if (client.i_batch < (int) i || client.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                //printf("client %d, seq %d, token %d, pos %d, batch %d\n",
                //        client.id, client.seq_id, client.sampled, client.n_decoded, client.i_batch);

                const llama_token id = llama_sampling_sample(client.ctx_sampling, ctx, NULL, client.i_batch - i);

                llama_sampling_accept(client.ctx_sampling, ctx, id, true);

                if (client.n_decoded == 1) {
                    // start measuring generation time after the first token to make sure all concurrent clients
                    // have their prompt already processed
                    client.t_start_gen = ggml_time_us();
                }

                const std::string token_str = llama_token_to_piece(ctx, id);

                // printf("client %d, seq %d, token %d, pos %d, batch %d: %s\n",
                //         client.id, client.seq_id, id, client.n_decoded, client.i_batch, token_str.c_str());

                if (params.custom_prompts_on) {
                    // is it an end of generation - are we done?
                    if (!client.valid_reply && (token_str.find('{') != std::string::npos)) {
                        // accepted answers start wiht '{' character
                        client.valid_reply = true;
                    }

                    if (client.valid_reply) {
                        // batched output
                        client.response += token_str;
                    } 

                    // tracking for next decode iteration
                    client.sampled = id;

                    // force end of output if we have a valid JSON reply or never detected
                    // a '{' character during decoding phase
                    if ((token_str.find('}') != std::string::npos) ||
                        ((client.n_decoded > 128) && !client.valid_reply)) {
                        // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                        llama_kv_cache_seq_rm(ctx, client.id + 1, -1, -1);
                        llama_kv_cache_seq_cp(ctx, 0, client.id + 1, -1, -1);

                        const auto t_main_end = ggml_time_us();

                        LOG_TEE("\nClient %3d, seq %3d/%3d, prompt %4d t, response %4d t, time %5.2f s, speed %5.2f t/s, cache miss %d "
                                "\nInput:    %s\nResponse: \n%s\n\n",
                                client.id, (client.seq_id + 1), n_seq, client.n_prompt, client.n_decoded,
                                (t_main_end - client.t_start_prompt) / 1e6,
                                (double) (client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6,
                                n_cache_miss,
                                ::trim(client.input).c_str(),
                                ::trim(client.response).c_str());

                        n_total_prompt += client.n_prompt;
                        n_total_gen    += client.n_decoded;

                        // signal we are done with this client
                        client.seq_id = -1;
                        client.valid_reply = false;
                    }

                } else {
                    // no custom promts
        
                    client.response += token_str;
                    client.sampled = id;
        
                    // printf("client %d, seq %d, token %d, pos %d, batch %d: %s\n",
                    //         client.id, client.seq_id, id, client.n_decoded, client.i_batch, token_str.c_str());
        
                    if (client.n_decoded > 2 &&
                            (llama_token_is_eog(model, id) ||
                             ((params.n_predict > 0) && ((client.n_decoded + client.n_prompt) >= params.n_predict)))) {
#if 0 // no reverse prompt for our scenario                                
                        // basic reverse prompt
                        const size_t pos = client.response.find("User:");
                        if (pos != std::string::npos) {
                            client.response = client.response.substr(0, pos);
                        }
#endif
                        // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                        llama_kv_cache_seq_rm(ctx, client.id + 1, -1, -1);
                        llama_kv_cache_seq_cp(ctx, 0, client.id + 1, -1, -1);
        
                        const auto t_main_end = ggml_time_us();
        
                        LOG_TEE("\nClient %3d, seq %3d/%3d, prompt %4d t, response %4d t, time %5.2f s, speed %5.2f t/s, cache miss %d \nInput:    %s\nResponse: %s\n\n",
                                client.id, (client.seq_id + 1), n_seq, client.n_prompt, client.n_decoded,
                                (t_main_end - client.t_start_prompt) / 1e6,
                                (double) (client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6,
                                n_cache_miss,
                                ::trim(client.input).c_str(),
                                ::trim(client.response).c_str());
        
                        n_total_prompt += client.n_prompt;
                        n_total_gen    += client.n_decoded;
        
                        client.seq_id = -1;
                    }                    
                }

                client.i_batch = -1;
            }
        }
    }

    const auto t_main_end = ggml_time_us();
    printf("\n\n total elapsed time %7.2fsec\n", (double) (t_main_end - t_main_start) / (1000. * 1000.));

    LOG_TEE("\n\nSummary stats:%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n", 
        __func__, n_clients, n_seq, cont_batching, n_tokens_system);
    if (params.custom_prompts_on) {
        params.prompt_file = params.custom_p_file;
    } else if (params.prompt_file.empty()) {
        params.prompt_file = "used built-in defaults";
    }
    LOG_TEE("External prompt file: %s\n", params.prompt_file.c_str());
    LOG_TEE("Model and path used:  %s\n\n", params.model.c_str());
    LOG_TEE("Number of threads:    %6d\n", params.n_threads);
    LOG_TEE("Number of prompts:    %6d\n", params.n_sequences);
    LOG_TEE("Number of clients:    %6d\n\n", n_clients);

    LOG_TEE("Total prompt tokens:  %6d, speed: %5.2f t/s\n", n_total_prompt, (double) (n_total_prompt              ) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Total gen tokens:     %6d, speed: %5.2f t/s\n", n_total_gen,    (double) (n_total_gen                 ) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Total speed (AVG):    %6s  speed: %5.2f t/s\n", "",             (double) (n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Cache misses:         %6d\n", n_cache_miss);

    LOG_TEE("\n");

    llama_print_timings(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

#ifdef GGML_TENSOR_OP_PERF
    print_tensor_op_perf_data();
#endif // GGML_TENSOR_OP_PERF

    fprintf(stderr, "\n\n");

    return 0;
}
