#pragma once

#include "llama.h"
#include "log.h"

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <filesystem>

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
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#endif // WIN32

struct xbapp_params {
    uint32_t seed                      = 42;   // RNG sampling seed - default was 0xFFFFFFFF
    uint32_t n_ctx                     = 1536; // context size (max of n_len + n_seqlen)
    int32_t n_len                      = 1024; // max length of the prompt including the system prompt
    int32_t n_threads                  = 8;
    int32_t n_batch                    = 512;  // size for a single batch (could be as large as prompt size)
    int32_t n_ngl                      = 0;    // number of layers offloaded to GPU
    int32_t n_seqlen                   = 128;  // max sequence length to generate
    int32_t verbose_level              = 0;    // verbose level (0 - none, 1 - info, 2 - warn, 3 - error, 4 - debug)
    std::string model_path             = "";   // model path
    std::string prompt                 = "";
    std::string custom_p_file          = "custom_prompts.txt";  // custom prompts input file
    std::string custom_template_prompt = "";
    std::string pfx_shared             = "";    // shared prompt for prefix cache (or prompt cache)
    std::string pfx_file               = "";    // file name for prefix cache
    bool pfc_mode                      = false; // prefix cache mode
    bool first_prompt                  = true;  // indicate first time through
    bool openmp                        = false; // true when openmp is present
    bool verbose_extra                 = false; // true for extra llama logging (i.e. debug messages)
    bool process_affinity              = false; // true if set process affinity is enabled
    ggml_log_level log_level           = (ggml_log_level)0;
};


int slm_inference(xbapp_params& params);
int slm_init(xbapp_params& params);
void slm_terminate();

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
