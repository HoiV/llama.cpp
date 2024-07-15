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

struct gpt_params {
    uint32_t seed = 42;   // RNG seed - default was 0xFFFFFFFF
    uint32_t n_ctx                     = 1536; // context size
    int32_t n_len                      = 1532; // total length of the sequence including the prompt
    int32_t n_threads                  = 12;
    int32_t n_batch                    = 512;  // size for a single batch
    std::string model                  = "";   // model path
    std::string prompt                 = "";
    std::string custom_p_file          = "custom_prompts.txt";  // custom prompts input file
    std::string custom_template_prompt = "";
    std::string pfx_shared             = "";   // shared prompt for prefix cache
    std::string pfx_file               = "";   // file name for prefix cache
    bool pfc_mode                      = false; // prefix cache mode
    bool first_prompt                  = true; // indicate first time through
};


int slm_inference(gpt_params& params);
int slm_init(gpt_params& params);
void slm_terminate();
