/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    LlmService.cpp

Abstract:

    This module implements the LLM service communication for AIMX.
    Handles HTTP communication with the LLM backend service.

Author:

    Rupo Zhang (rizhang) 03/20/2025

--*/

#include "pch.h"
#include "RAGBuilder.h"
#include "RAGHelper.h"

#include <codecvt>
#include <locale>
#include <sstream>
#include <filesystem>

class LlmService {
public:
    LlmService();
    ~LlmService();

    // RAG functionality
    bool
    InitializeRagDatabase();
    
    bool
    IsRagInitialized() const;
    
private:
    bool m_ragInitialized;
    model_params m_embeddingParams;
};

/*++

Routine Description:

    Constructor for LlmService class. Initializes the LLM service.

Arguments:

    None.

Return Value:

    None.

--*/
LlmService::LlmService()
    : m_ragInitialized(false)
{
    // Try to initialize the RAG database if it exists
    InitializeRagDatabase();
}

/*++

Routine Description:

    Destructor for LlmService class. Performs cleanup.

Arguments:

    None.

Return Value:

    None.

--*/
LlmService::~LlmService()
{
    // Call the cleanup function from RAGHelper.cpp
    CleanupRagResources();
}

/*++

Routine Description:

    Initializes the RAG database for context retrieval.

Arguments:

    None.

Return Value:

    bool indicating success or failure of the initialization.

--*/
bool
LlmService::InitializeRagDatabase()
{
    Debug::Log("Initializing RAG database");
    
    if (m_ragInitialized)
    {
        Debug::Log("RAG database already initialized");
        return true;
    }
    
    try
    {
        // Check if the necessary files exist in the ragdb directory
        std::filesystem::path ragIndexPath = "./ragdb/rag_index.bin";
        std::filesystem::path chunksJsonPath = "./ragdb/chunks.json";
        
        if (!std::filesystem::exists(ragIndexPath) || !std::filesystem::exists(chunksJsonPath))
        {
            Debug::LogError("RAG database files not found: " + 
                          WideToUtf8(ragIndexPath.wstring()) + " or " + 
                          WideToUtf8(chunksJsonPath.wstring()));
            return false;
        }
        
        // Initialize embedding model parameters
        m_embeddingParams.model_name = "./models/all-minilm-l6-v2_f32.gguf";
        m_embeddingParams.n_ctx = 512;
        
        // Initialize the RAG database using the RAGHelper function
        if (!RAG_initialize(m_embeddingParams))
        {
            Debug::LogError("Failed to initialize RAG database");
            return false;
        }
        
        m_ragInitialized = true;
        Debug::Log("RAG database initialized successfully");
        return true;
    }
    catch (const std::exception& e)
    {
        Debug::LogError("Exception during RAG initialization: " + std::string(e.what()));
        return false;
    }
}

/*++

Routine Description:

    Checks if the RAG database is initialized.

Arguments:

    None.

Return Value:

    bool indicating whether the RAG database is initialized.

--*/
bool
LlmService::IsRagInitialized() const
{
    return m_ragInitialized;
}

int main(int argc, char *argv[]) {
    LlmService* llmService = new LlmService;

    if (llmService == NULL) {
        return -1;
    }

    try
    {
        printf("Initializing RAG builder\n");

        model_params params;
        params.model_name = "./models/all-minilm-l6-v2_f32.gguf";
        params.n_ctx = 512;
        params.n_dim = 384;

        RAGBuilder ragbuilder;
        ragbuilder.Initialize(params, "./ragDB", 128);
        printf("Processing documents...\n");
        ragbuilder.ProcessDocumentDirectory("./docs");
        printf("Building RAG database\n");
        ragbuilder.BuildDatabase();

        bool ragInitialized = llmService->InitializeRagDatabase();
        if (ragInitialized) {
            printf("RAG database initialized successfully");

        } else {
                printf("RAG database initialization failed, continuing without RAG functionality");
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "RAG initialization exception: " + std::string(e.what()) << std::endl;
    }
}