#pragma once

#include "llama.h"
#include "common.h"
#include "console.h"
#include "log.h"

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <filesystem>
#include <algorithm>

#ifdef _WIN32
#if !defined WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#include <stdint.h>
#endif

typedef volatile LONG atomic_int;
inline void atomic_store(atomic_int * ptr, LONG val) {
    InterlockedExchange(ptr, val);
}
inline LONG atomic_load(atomic_int * ptr) {
    return InterlockedCompareExchange(ptr, 0, 0);
}

#define DEFAULT_SYSTEM_PROMPT \
    "A chat between a curious human and an artificial intelligence assistant. " \
    "The assistant gives helpful, detailed, and polite answers to the " \
    "human's questions."

//struct clip_ctx;
struct gpt_params;
struct llama_context;
struct llama_model;

enum Role {
    ROLE_USER,
    ROLE_ASSISTANT,
    ROLE_SYSTEM,
    ROLE_UNKNOWN,
};

enum SpecialToken {
    IMAGE_PLACEHOLDER_TOKEN = -31337,
};

extern bool g_manual_mode;
extern bool g_said_something;
extern char g_last_printed_char;
//extern clip_ctx *g_clip;
extern enum Role g_role;
extern gpt_params g_params;
extern int g_system_prompt_tokens;
extern llama_context *g_ctx;
extern llama_model *g_model;
extern std::vector<int> g_history;
extern atomic_int g_got_sigint;

int main(int, char **);

bool eval_string(std::string_view, bool, bool);
bool eval_token(int);
bool eval_tokens(std::vector<int>);
bool handle_command(const char *);
bool is_base_model();
bool out_of_context(int);
char *on_hint(const char *, const char **, const char **);
const char *get_role_color(enum Role);
const char *get_role_name(enum Role);
enum Role cycle_role(enum Role);
enum Role get_next_role(enum Role);
int tokens_used(void);
std::string token_to_piece(const llama_context *, int, bool);
void adjust_stacks(int, int);
void clear_ephemeral(void);
void ensure_newline();
void err(const char *, ...);
void fix_stacks(void);
void logo(char **);
void on_clear(const std::vector<std::string> &);
void on_context(const std::vector<std::string> &);
void on_dump(const std::vector<std::string> &);
void on_forget(const std::vector<std::string> &);
void on_help(const std::vector<std::string> &);
void on_manual(const std::vector<std::string> &);
void on_pop(const std::vector<std::string> &);
void on_push(const std::vector<std::string> &);
void on_stack(const std::vector<std::string> &);
void on_undo(const std::vector<std::string> &);
void on_upload(const std::vector<std::string> &);
void on_quit(const std::vector<std::string> &);
void print(const std::string_view &);
void print_ephemeral(const std::string_view &);
void record_undo(void);
void repl();
void rewind(int);

#define RESET "\x1b[0m"
#define BOLD "\x1b[1m"
#define FAINT "\x1b[2m"
#define UNBOLD "\x1b[22m"
#define RED "\x1b[31m"
#define GREEN "\x1b[32m"
#define MAGENTA "\x1b[35m"
#define YELLOW "\x1b[33m"
#define CYAN "\x1b[36m"
#define UNFOREGROUND "\x1b[39m"
#define BRIGHT_BLACK "\x1b[90m"
#define BRIGHT_RED "\x1b[91m"
#define BRIGHT_GREEN "\x1b[92m"
#define CLEAR_FORWARD "\x1b[K"

// Many llama.cpp APIs take boolean parameters at the end. Please favor
// passing these constants as arguments instead, for better readability

#define ADD_SPECIAL true
#define DONT_ADD_SPECIAL false

#define PARSE_SPECIAL true
#define DONT_PARSE_SPECIAL false

#define ADD_ASSISTANT true
#define DONT_ADD_ASSISTANT false

#define APPLY_GRAMMAR true
#define DONT_APPLY_GRAMMAR false

#define RENDER_SPECIAL_TOKENS true
#define DONT_RENDER_SPECIAL_TOKENS false

// same with llama_chat_message, but uses std::string
struct llama_chat_msg {
    std::string role;
    std::string content;
};

// CPP wrapper for llama_chat_apply_template
// If the built-in template is not supported, we default to chatml
// If the custom "tmpl" is not supported, we throw an error
std::string llama_chat_apply_template(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & chat,
        bool add_ass);

// Format single message, while taking into account the position of that message in chat history
std::string llama_chat_format_single(const struct llama_model * model,
        const std::string & tmpl,
        const std::vector<llama_chat_msg> & past_msg,
        const llama_chat_msg & new_msg,
        bool add_ass);

// Returns an example of formatted chat
std::string llama_chat_format_example(const struct llama_model * model,
        const std::string & tmpl);

// replaces multiple isspace() with one space and trims result
std::string collapse(const std::string_view &input);

std::string basename(const std::string_view &path);
