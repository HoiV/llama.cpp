#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <unordered_map>

#include "hnswlib/hnswlib.h"

#if !defined WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#ifdef LLM_INFER_SHARED
#    if defined(_WIN32)
#        ifdef LLM_INFER_DLL
#            pragma message("Build llm-infer DLL APIs")
#            define LLM_INFER_API __declspec(dllexport)
#        else
#            pragma message("Build llm-infer clients")
#            define LLM_INFER_API __declspec(dllimport)
#        endif
#    endif
#else
#    define LLM_INFER_API
#endif

struct model_params {
    uint32_t seed                      = 42;   // RNG seed - default was 0xFFFFFFFF
    uint32_t n_ctx                     = 2048; // context size
    int32_t n_len                      = 1532; // total length of the sequence including the prompt
    int32_t n_threads                  = 8;
    int32_t n_batch                    = 512;  // size for a single batch
    int32_t n_dim                      = 384;  // projection dim for the model (only for embeddings)
    int32_t chunk_size                 = 128;  // max chunk size for RAG partitioning
    std::string model_name             = "";   // model path
    std::string prompt                 = "";
    std::string reply                  = "";
    int total_llm_tokens_generated     = 0;
    int verbose                        = 0;
    int streaming_reply                = 0; // streaming mode instead of full reply at once
    int force_cpu_mode                 = 0; // default is to use GPU if there is one present

    std::string custom_p_file          = "custom_prompts.txt";  // custom prompts input file
    std::string custom_template_prompt = "";
    std::string pfx_shared             = "";   // shared prompt for prefix cache
    std::string pfx_file               = "";   // file name for prefix cache
    bool pfc_mode                      = false; // prefix cache mode
    bool first_prompt                  = true; // indicate first time through
    bool save_llm_state                = false;
};

LLM_INFER_API bool llm_initialize(model_params& params);
LLM_INFER_API bool llm_inference(model_params& params);
LLM_INFER_API void llm_terminate(const model_params&);

typedef int32_t llama_token;
struct chunk {
    // filename
    std::string filename;
    // original text data
    std::string textdata;
    // tokens
    std::vector<llama_token> tokens;
    // embeddings
    std::vector<float> embeddings;
};

struct rag_entry {
    // index for this entry
    int id;
    // filename
    std::string filename;
    // text data
    std::string textdata;
};

LLM_INFER_API bool embed_initialize(model_params & params);
LLM_INFER_API bool embed_encode_batch(const model_params & params, std::vector<chunk> & chunks);
LLM_INFER_API bool embed_encode_single(const model_params & params, const std::string& query, std::vector<float> &);
LLM_INFER_API void embed_terminate();
