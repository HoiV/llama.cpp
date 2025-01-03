// -*- mode:c++;indent-tabs-mode:nil;c-basic-offset:4;coding:utf-8 -*-
// vi: set et ft=cpp ts=4 sts=4 sw=4 fenc=utf-8 :vi
//
// Copyright 2024 Mozilla Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "chatbot.h"

#ifdef GGML_USE_OPENMP
#include "omp.h"
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))

gpt_params g_params;
//clip_ctx *g_clip;
llama_model *g_model;
llama_context *g_ctx;
//pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;
//pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
//std::string g_listen_url;

bool g_has_ephemeral;
bool g_said_something;
char g_last_printed_char;
volatile atomic_int64 g_got_sigint;

void on_sigint(int sig) {
    g_got_sigint = 1;
}

bool is_empty(const char *s) {
    int c;
    while ((c = *s++))
        if (!isspace(c))
            return false;
    return true;
}

void print(const std::string_view &s) {
    for (char c : s) {
        g_last_printed_char = c;
        fputc(c, stdout);
        if (c == '\n')
            g_has_ephemeral = false;
    }
}

void ensure_newline() {
    if (g_last_printed_char != '\n')
        print("\n");
}

void err(const char *fmt, ...) {
    va_list ap;
    clear_ephemeral();
    ensure_newline();
    va_start(ap, fmt);
    fputs(BRIGHT_RED, stderr);
    vfprintf(stderr, fmt, ap);
    fputs(RESET "\n", stderr);
    va_end(ap);
}

// replaces multiple isspace() with one space and trims result
std::string collapse(const std::string_view &input) {
    size_t start = 0;
    while (start < input.length() && std::isspace(input[start]))
        ++start;
    if (start == input.length())
        return "";
    size_t end = input.length() - 1;
    while (end > start && std::isspace(input[end]))
        --end;
    std::string result;
    result.reserve(end - start + 1);
    bool lastWasSpace = false;
    for (size_t i = start; i <= end; ++i) {
        if (std::isspace(input[i])) {
            if (!lastWasSpace) {
                result += ' ';
                lastWasSpace = true;
            }
        } else {
            result += input[i];
            lastWasSpace = false;
        }
    }
    return result;
}

std::string basename(const std::string_view &path) {
    size_t i, e;
    if ((e = path.size())) {
        while (e > 1 && path[e - 1] == '/')
            --e;
        i = e - 1;
        while (i && path[i - 1] != '/')
            --i;
        return std::string(path.substr(i, e - i));
    } else {
        return ".";
    }
}

void print_ephemeral(const std::string_view &description) {
    fprintf(stderr, " " BRIGHT_BLACK "%.*s" UNFOREGROUND "\r", (int)description.size(),
            description.data());
    g_has_ephemeral = true;
}

void clear_ephemeral(void) {
    if (g_has_ephemeral) {
        fprintf(stderr, CLEAR_FORWARD);
        g_has_ephemeral = false;
    }
}

int chatbot_token_eot(llama_model *model) {
    llama_token eot = llama_token_eot(model);
    if (eot != -1)
        return eot;
    return llama_token_eos(model);
}

std::string token_to_piece(const struct llama_context *ctx, llama_token token, bool special) {
    if (token == IMAGE_PLACEHOLDER_TOKEN)
        return "⁑";
    return llama_token_to_piece(ctx, token, special);
}

bool out_of_context(int extra) {
    err("error: ran out of context window at %d tokens\n"
        "consider passing `-c %d` at startup for the maximum\n"
        "you can free up more space using /forget or /clear",
        tokens_used() + extra, llama_n_ctx_train(g_model));
    return false;
}

// Xbox-B612
std::string llama_chat_apply_template(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & msgs,
        bool add_ass) {
    int alloc_size = 0;
    bool fallback = false; // indicate if we must fallback to default chatml
    std::vector<llama_chat_message> chat;
    for (auto & msg : msgs) {
        chat.push_back({msg.role.c_str(), msg.content.c_str()});
        alloc_size += (msg.role.size() + msg.content.size()) * 1.25;
    }

    const char * ptr_tmpl = tmpl.empty() ? nullptr : tmpl.c_str();
    std::vector<char> buf(alloc_size);

    // run the first time to get the total output length
    int32_t res = llama_chat_apply_template(model, ptr_tmpl, chat.data(), chat.size(), add_ass, buf.data(), buf.size());

    // error: chat template is not supported
    if (res < 0) {
        if (ptr_tmpl != nullptr) {
            // if the custom "tmpl" is not supported, we throw an error
            // this is a bit redundant (for good), since we're not sure if user validated the custom template with llama_chat_verify_template()
            throw std::runtime_error("this custom template is not supported");
        } else {
            // If the built-in template is not supported, we default to chatml
            res = llama_chat_apply_template(nullptr, "chatml", chat.data(), chat.size(), add_ass, buf.data(), buf.size());
            fallback = true;
        }
    }

    // if it turns out that our buffer is too small, we resize it
    if ((size_t) res > buf.size()) {
        buf.resize(res);
        res = llama_chat_apply_template(
            fallback ? nullptr : model,
            fallback ? "chatml" : ptr_tmpl,
            chat.data(), chat.size(), add_ass, buf.data(), buf.size());
    }

    std::string formatted_chat(buf.data(), res);
    return formatted_chat;
}

std::string llama_chat_format_single(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & past_msg,
        const llama_chat_msg & new_msg,
        bool add_ass) {
    std::ostringstream ss;
    auto fmt_past_msg = past_msg.empty() ? "" : llama_chat_apply_template(model, tmpl, past_msg, false);
    std::vector<llama_chat_msg> chat_new(past_msg);
    // if the past_msg ends with a newline, we must preserve it in the formatted version
    if (add_ass && !fmt_past_msg.empty() && fmt_past_msg.back() == '\n') {
        ss << "\n";
    };
    // format chat with new_msg
    chat_new.push_back(new_msg);
    auto fmt_new_msg = llama_chat_apply_template(model, tmpl, chat_new, add_ass);
    // get the diff part
    ss << fmt_new_msg.substr(fmt_past_msg.size(), fmt_new_msg.size() - fmt_past_msg.size());
    return ss.str();
}

std::string llama_chat_format_example(const struct llama_model * model,
        const std::string & tmpl) {
    std::vector<llama_chat_msg> msgs = {
        {"system",    "You are a helpful assistant"},
        {"user",      "Hello"},
        {"assistant", "Hi there"},
        {"user",      "How are you?"},
    };
    return llama_chat_apply_template(model, tmpl, msgs, true);
}
// End of Xbox-B612

bool eval_tokens(std::vector<llama_token> tokens) {
    int N = (int)tokens.size();
    if ((tokens_used() + N) > llama_n_ctx(g_ctx))
        return out_of_context(N);
    for (int i = 0; i < N; i += g_params.n_batch) {
        if (g_got_sigint) {
            g_got_sigint = false;
            clear_ephemeral();
            return false;
        }
        if (N > g_params.n_batch)
            printf("loading prompt %d...", (int)((double)i / N * 100));
        int n_eval = (int)tokens.size() - i;
        if (n_eval > g_params.n_batch)
            n_eval = g_params.n_batch;
        if (llama_decode(g_ctx, llama_batch_get_one(&tokens[i], n_eval, tokens_used(), 0)))
            return out_of_context(n_eval);
        g_history.insert(g_history.end(), tokens.begin() + i, tokens.begin() + i + n_eval);
    }
    clear_ephemeral();
    // this function is what computes /stats. we need to call it now
    // since llama_decode() kicks the can down the road to functions
    // like llama_sampling_sample(). that is bad because the chatbot
    // returns control to the repl rather than sampling when loading
    // system and image prompts.
    llama_synchronize(g_ctx);
    return true;
}

#if 0
bool eval_image_embed(const llava_image_embed *image_embed) {
    int N = image_embed->n_image_pos;
    if (tokens_used() + N > llama_n_ctx(g_ctx))
        return out_of_context(N);
    int n_embd = llama_n_embd(llama_get_model(g_ctx));
    for (int i = 0; i < N; i += g_params.n_batch) {
        if (g_got_sigint) {
            g_got_sigint = false;
            clear_ephemeral();
            return false;
        }
        if (N > g_params.n_batch)
            print_ephemeral(format("loading image %d%%...", (int)((double)i / N * 100)));
        int n_eval = N - i;
        if (n_eval > g_params.n_batch)
            n_eval = g_params.n_batch;
        llama_batch batch = {
            .n_tokens = n_eval,
            .embd = image_embed->embed + i * n_embd,
            .all_pos_0 = tokens_used(),
            .all_pos_1 = 1,
            .all_seq_id = 0,
        };
        if (llama_decode(g_ctx, batch))
            return out_of_context(n_eval);
        for (int i = 0; i < n_eval; ++i)
            g_history.push_back(IMAGE_PLACEHOLDER_TOKEN);
    }
    clear_ephemeral();
    llama_synchronize(g_ctx);
    return true;
}

bool eval_image(const std::string_view binary) {
    unassert(g_clip);
    llava_image_embed *image_embed;
    print_ephemeral("analyzing image...");
    image_embed = llava_image_embed_make_with_bytes(
        g_clip, FLAG_threads_batch, (const unsigned char *)binary.data(), binary.size());
    clear_ephemeral();
    if (!image_embed) {
        err("failed to load image");
        return false;
    }
    bool ok = eval_image_embed(image_embed);
    llava_image_embed_free(image_embed);
    return ok;
}
#endif

bool eval_token(int id) {
    return eval_tokens({id});
}

void on_quit(const std::vector<std::string> &args) {
    print_ephemeral("quitting chatbot...");
    clear_ephemeral();

    print_ephemeral("freeing context...");
    llama_free(g_ctx);
    clear_ephemeral();

    print_ephemeral("freeing model...");
    llama_free_model(g_model);
    clear_ephemeral();

    print_ephemeral("freeing backend...");
    llama_backend_free();
    clear_ephemeral();

    exit(0);
}

void chat_loop() {
    // setup conversation
    if (llama_should_add_bos_token(g_model)) {
        print_ephemeral("loading bos token...");
        eval_token(llama_token_bos(g_model));
    }
    record_undo();

    // make base models have no system prompt by default
    if (is_base_model() && g_params.prompt == DEFAULT_SYSTEM_PROMPT)
        g_params.prompt = "";

    // setup system prompt
    if (!g_params.prompt.empty()) {
        print_ephemeral("loading system prompt...");
        std::string msg;
        if (is_base_model()) {
            msg = g_params.prompt;
        } else {
            std::vector<llama_chat_msg> chat = {{"system", g_params.prompt}};
            msg = llama_chat_apply_template(g_model, g_params.chat_template, chat,
                                            DONT_ADD_ASSISTANT);
        }
        if (!eval_tokens(llama_tokenize(g_model, msg, DONT_ADD_SPECIAL, PARSE_SPECIAL)))
            exit(6);
        llama_synchronize(g_ctx);
        g_system_prompt_tokens = tokens_used();
        clear_ephemeral();
        if (g_params.display_prompt)
            printf("%s\n", g_params.special ? msg.c_str() : g_params.prompt.c_str());
    }

    // perform important setup
    llama_sampling_context *sampler = llama_sampling_init(g_params.sparams);
    //signal(SIGINT, on_sigint);

    // run chatbot
    for (;;) {
        record_undo();
        _write(1, get_role_color(g_role), strlen(get_role_color(g_role)));

        // color user input only
        console::set_display(console::user_input);

        std::string line;
        std::string buffer;
        bool another_line = true;
        do {
            another_line = console::readline(line, /* g_params.multiline_input */ false);
            buffer += line;
        } while (another_line);

        // done taking input, reset color
        console::set_display(console::reset);

        _write(1, RESET, strlen(RESET));
        g_last_printed_char = '\n';
        if (buffer.empty()) {
            if (g_got_sigint)
                ensure_newline();
            break;
        }
        if (!is_base_model() && buffer.empty()) {
            if (g_manual_mode) {
                g_role = cycle_role(g_role);
                _write(1, "\033[F", 3);
            }
            continue;
        }
        g_said_something = true;
        if (handle_command(buffer.c_str())) {
            continue;
        }
        bool add_assi = !g_manual_mode;
        int tokens_used_before = tokens_used();
        std::string msg;
        if (is_base_model()) {
            msg = buffer;
        } else {
            std::vector<llama_chat_msg> chat = {{get_role_name(g_role), buffer}};
            msg = llama_chat_apply_template(g_model, g_params.chat_template, chat, add_assi);
        }
        if (!eval_tokens(llama_tokenize(g_model, msg, DONT_ADD_SPECIAL, PARSE_SPECIAL))) {
            rewind(tokens_used_before);
            continue;
        }
        if (g_manual_mode) {
            g_role = get_next_role(g_role);
            continue;
        }
        for (;;) {
            if (g_got_sigint) {
                eval_token(chatbot_token_eot(g_model));
                break;
            }
            llama_token id = llama_sampling_sample(sampler, g_ctx, NULL);
            llama_sampling_accept(sampler, g_ctx, id, APPLY_GRAMMAR);
            if (!eval_token(id))
                break;
            if (llama_token_is_eog(g_model, id))
                break;
            std::string s = token_to_piece(g_ctx, id, g_params.special);
            print(s);
            fflush(stdout);
        }
        g_got_sigint = 0;
        std::string s;
        print(s);
        ensure_newline();
    }

    // cleanup resources
    llama_sampling_free(sampler);
}

bool is_base_model() {
    // check if user explicitly passed --chat-template flag
    if (!g_params.chat_template.empty())
        return false;

    // check if gguf metadata has chat template. this should always be
    // present for "instruct" models, and never specified on base ones
    return llama_model_meta_val_str(g_model, "tokenizer.chat_template", 0, 0) == -1;
}

int main(int argc, char **argv) {
    // g_params.simple_io and g_params.use_color
    console::init(true, true);

    // override defaults for some flags
    g_params.n_batch = 256; // for better progress indication
    g_params.sparams.temp = 0; // don't believe in randomness by default
    g_params.prompt = DEFAULT_SYSTEM_PROMPT;

    // parse flags (sadly initializes gpu support as side-effect)
    print_ephemeral("loading backend...");
    llama_backend_init();
    if (!gpt_params_parse(argc, argv, g_params)) { // also loads gpu module
        fprintf(stderr, "error: failed to parse flags\n");
        exit(1);
    }
    clear_ephemeral();

#ifdef GGML_USE_OPENMP
    g_params.n_threads = MIN(g_params.n_threads, omp_get_max_threads());
    if (g_params.use_omp) {
        printf("%s: OpenMP selected\n", __func__);
        ggml_select_omp();
    }
#endif

    if (g_params.use_proc_affinity) {
        common::xb_set_optimal_process_affinity(g_params.n_threads);
    }

    // setup logging
    // FLAG_log_disable = false;
    if (!g_params.verbosity)
        log_disable();

    print_ephemeral("loading model...");
    llama_model_params model_params = llama_model_params_from_gpt_params(g_params);
    g_model = llama_load_model_from_file(g_params.model.c_str(), model_params);
    clear_ephemeral();
    if (g_model == NULL) {
        fprintf(stderr, "... failed to load model '%s'\n", g_params.model.c_str());
        exit(2);
    }
    if (g_params.n_ctx <= 0 || g_params.n_ctx > llama_n_ctx_train(g_model))
        g_params.n_ctx = llama_n_ctx_train(g_model);
    if (g_params.n_ctx < g_params.n_batch)
        g_params.n_batch = g_params.n_ctx;

    if (g_params.verbosity != 0) {
        printf(BOLD "model" UNBOLD ":    %s\n", basename(g_params.model).c_str());
        if (is_base_model())
            printf(BOLD "mode" UNBOLD ":     RAW TEXT COMPLETION (base model)\n");
        printf("\n");
    }

    print_ephemeral("initializing context...");
    llama_context_params ctx_params = llama_context_params_from_gpt_params(g_params);
    g_ctx = llama_new_context_with_model(g_model, ctx_params);
    clear_ephemeral();
    if (!g_ctx) {
        fprintf(stderr, "error: failed to initialize context\n");
        exit(3);
    }

#if 0
    if (FLAG_mmproj) {
        printf("initializing vision model...");
        g_clip = clip_model_load(FLAG_mmproj, g_params.verbosity);
        if (!g_clip) {
            fprintf(stderr, "%s: failed to initialize clip image model\n", FLAG_mmproj);
            exit(4);
        }
    }
#endif

    chat_loop();

#if 0
    if (g_clip) {
        print_ephemeral("freeing vision model...");
        clip_free(g_clip);
        clear_ephemeral();
    }
#endif

    print_ephemeral("freeing context...");
    llama_free(g_ctx);
    clear_ephemeral();

    print_ephemeral("freeing model...");
    llama_free_model(g_model);
    clear_ephemeral();

    print_ephemeral("freeing backend...");
    llama_backend_free();
    clear_ephemeral();

    return 0;
}
