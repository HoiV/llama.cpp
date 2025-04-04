/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    RAGHelper.h

Abstract:

    This module defines the interface for the RAG helper functions.
    Provides functions for initializing and using the RAG database.
    Most of content are incorporated from the file:
    %SDXROOT%\xbox\gamecore\so2001\z-slmapp\llm-infer\include\llm-infer.h

Author:

    Rupo Zhang (rizhang) 03/22/2025

--*/

#pragma once

#include "pch.h"

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

    std::string custom_p_file          = "custom_prompts.txt";  // custom prompts input file
    std::string custom_template_prompt = "";
    std::string pfx_shared             = "";   // shared prompt for prefix cache
    std::string pfx_file               = "";   // file name for prefix cache
    bool pfc_mode                      = false; // prefix cache mode
    bool first_prompt                  = true; // indicate first time through
    bool save_llm_state                = false;
};

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

bool 
LoadLlmInferDll();

void
CleanupRagResources();

void 
ListDllExports(
    _In_ HMODULE hModule
    );

bool 
RAG_initialize(
    _In_ model_params & eparams
    );

std::vector<rag_entry>
retrieve_chunks(
    _In_ const model_params & eparams,
    _In_ const std::string& query, 
    _In_ int top_k = 3
    );

// 
// Below are the functions imported from the llm-infer.dll
//

bool
embed_initialize(
    _In_ model_params& params
    );

bool 
embed_encode_batch(
    _In_ const model_params& params,
    _Inout_ std::vector<chunk>& chunks
    );

bool 
embed_encode_single(
    _In_ const model_params& params,
    _In_ const std::string& query,
    _Out_ std::vector<float>& embeddings
    );

void 
embed_terminate();

