/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    RAGHelper.cpp

Abstract:

    This module implements helper functions for RAG functionality in AIMX.
    Provides interface with the llm-infer.dll library for RAG services.

Author:

    Rupo Zhang (rizhang) 03/22/2025

--*/

#include "pch.h"
#include "RAGHelper.h"

using json = nlohmann::json;

// Function pointer types for the llm-infer.dll functions
typedef bool (*EMBED_INITIALIZE_FUNC)(model_params&);
typedef bool (*EMBED_ENCODE_BATCH_FUNC)(const model_params&, std::vector<chunk>&);
typedef bool (*EMBED_ENCODE_SINGLE_FUNC)(const model_params&, const std::string&, std::vector<float>&);
typedef void (*EMBED_TERMINATE_FUNC)();

// Function pointers for dynamic loading
static HMODULE g_hLlmInferDll = NULL;
static EMBED_INITIALIZE_FUNC g_pfnEmbedInitialize = NULL;
static EMBED_ENCODE_BATCH_FUNC g_pfnEmbedEncodeBatch = NULL;
static EMBED_ENCODE_SINGLE_FUNC g_pfnEmbedEncodeSingle = NULL;
static EMBED_TERMINATE_FUNC g_pfnEmbedTerminate = NULL;

// Global state for RAG database
static bool g_bRagDbInitialized = false;
static std::vector<rag_entry> g_RagEntries;
static hnswlib::HierarchicalNSW<float>* g_pRagDB = nullptr;

/*++

Routine Description:

    Loads the llm-infer.dll and gets function pointers to required exported functions.

Arguments:

    None.

Return Value:

    bool indicating success or failure of the DLL loading.

--*/
bool 
LoadLlmInferDll(
    void
    )
{
    if (g_hLlmInferDll != NULL)
    {
        // Already loaded
        return true;
    }
    
    Debug::Log("Loading llm-infer.dll...");
    
    // First try the current directory
    g_hLlmInferDll = LoadLibraryW(L"llm-infer.dll");
    
    // If that fails, try looking in application directory
    if (g_hLlmInferDll == NULL)
    {
        WCHAR modulePath[MAX_PATH];
        if (GetModuleFileNameW(NULL, modulePath, MAX_PATH))
        {
            // Get the directory
            WCHAR* lastSlash = wcsrchr(modulePath, L'\\');
            if (lastSlash)
            {
                *(lastSlash + 1) = L'\0'; // Truncate to directory
                std::wstring dllPath = std::wstring(modulePath) + L"llm-infer.dll";
                Debug::Log("Trying to load from application directory: " + WideToUtf8(dllPath));
                g_hLlmInferDll = LoadLibraryW(dllPath.c_str());
            }
        }
    }
    
    // If still not loaded, check standard paths
    if (g_hLlmInferDll == NULL)
    {
        // Check Windows directory
        WCHAR winDir[MAX_PATH];
        if (GetWindowsDirectoryW(winDir, MAX_PATH))
        {
            std::wstring dllPath = std::wstring(winDir) + L"\\System32\\llm-infer.dll";
            Debug::Log("Trying to load from System32: " + WideToUtf8(dllPath));
            g_hLlmInferDll = LoadLibraryW(dllPath.c_str());
        }
    }
    
    if (g_hLlmInferDll == NULL)
    {
        Debug::LogError("Failed to load llm-infer.dll. Error: " + std::to_string(GetLastError()));
        return false;
    }
    
    // Get function pointers with detailed error reporting - only for the functions we need
    g_pfnEmbedInitialize = (EMBED_INITIALIZE_FUNC)GetProcAddress(g_hLlmInferDll, "embed_initialize");
    if (g_pfnEmbedInitialize == NULL) 
    {
        Debug::LogError("Failed to get embed_initialize function. Error: " + std::to_string(GetLastError()));
    }
    
    g_pfnEmbedEncodeBatch = (EMBED_ENCODE_BATCH_FUNC)GetProcAddress(g_hLlmInferDll, "embed_encode_batch");
    if (g_pfnEmbedEncodeBatch == NULL) 
    {
        Debug::LogError("Failed to get embed_encode_batch function. Error: " + std::to_string(GetLastError()));
    }
    
    g_pfnEmbedEncodeSingle = (EMBED_ENCODE_SINGLE_FUNC)GetProcAddress(g_hLlmInferDll, "embed_encode_single");
    if (g_pfnEmbedEncodeSingle == NULL) 
    {
        Debug::LogError("Failed to get embed_encode_single function. Error: " + std::to_string(GetLastError()));
    }
    
    g_pfnEmbedTerminate = (EMBED_TERMINATE_FUNC)GetProcAddress(g_hLlmInferDll, "embed_terminate");
    if (g_pfnEmbedTerminate == NULL) 
    {
        Debug::LogError("Failed to get embed_terminate function. Error: " + std::to_string(GetLastError()));
    }
    
    // Check if all function pointers are valid
    if (g_pfnEmbedInitialize == NULL || g_pfnEmbedEncodeBatch == NULL || 
        g_pfnEmbedEncodeSingle == NULL || g_pfnEmbedTerminate == NULL)
    {
        // List available exports from the DLL for diagnostic purposes
        Debug::LogError("Failed to get function addresses from llm-infer.dll");
        ListDllExports(g_hLlmInferDll);
        
        FreeLibrary(g_hLlmInferDll);
        g_hLlmInferDll = NULL;
        return false;
    }
    
    Debug::Log("Successfully loaded llm-infer.dll and retrieved function pointers");
    return true;
}

/*++

Routine Description:

    Lists all exported functions from a DLL for diagnostic purposes.

Arguments:

    hModule - Handle to the loaded DLL module.

Return Value:

    None.

--*/
void 
ListDllExports(
    _In_ HMODULE hModule
    )
{
    if (!hModule) 
    {
        return;
    }
    
    Debug::Log("Attempting to list exports from DLL...");
    
    // Get DOS Header
    PIMAGE_DOS_HEADER pDosHeader = (PIMAGE_DOS_HEADER)hModule;
    if (pDosHeader->e_magic != IMAGE_DOS_SIGNATURE)
    {
        Debug::LogError("Invalid DOS header");
        return;
    }
    
    // Get NT Headers
    PIMAGE_NT_HEADERS pNtHeaders = (PIMAGE_NT_HEADERS)((BYTE*)hModule + pDosHeader->e_lfanew);
    if (pNtHeaders->Signature != IMAGE_NT_SIGNATURE)
    {
        Debug::LogError("Invalid NT header");
        return;
    }
    
    // Get export directory
    DWORD exportDirRVA = pNtHeaders->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress;
    if (exportDirRVA == 0)
    {
        Debug::LogError("No export directory found");
        return;
    }
    
    PIMAGE_EXPORT_DIRECTORY pExportDir = (PIMAGE_EXPORT_DIRECTORY)((BYTE*)hModule + exportDirRVA);
    PDWORD pdwNames = (PDWORD)((BYTE*)hModule + pExportDir->AddressOfNames);
    
    Debug::Log("Found " + std::to_string(pExportDir->NumberOfNames) + " named exports:");
    
    for (DWORD i = 0; i < pExportDir->NumberOfNames; i++)
    {
        std::string functionName = (char*)((BYTE*)hModule + pdwNames[i]);
        Debug::Log("Export #" + std::to_string(i) + ": " + functionName);
    }
}

/*++

Routine Description:

    Cleans up resources used by the RAG helper including freeing the loaded DLL.

Arguments:

    None.

Return Value:

    None.

--*/
void 
CleanupRagResources(
    void
    )
{
    // Clean up RAG if initialized
    if (g_bRagDbInitialized && g_pfnEmbedTerminate != NULL)
    {
        g_pfnEmbedTerminate();
        g_bRagDbInitialized = false;
    }
    
    // Free the DLL
    if (g_hLlmInferDll != NULL)
    {
        FreeLibrary(g_hLlmInferDll);
        g_hLlmInferDll = NULL;
        g_pfnEmbedInitialize = NULL;
        g_pfnEmbedEncodeBatch = NULL;
        g_pfnEmbedEncodeSingle = NULL;
        g_pfnEmbedTerminate = NULL;
    }
    
    // Free any allocated resources
    if (g_pRagDB != nullptr) 
    {
        delete g_pRagDB;
        g_pRagDB = nullptr;
    }
}

/*++

Routine Description:

    Initializes the RAG database for context retrieval.
    This is called by LlmService to set up the RAG database.

Arguments:

    eparams - Embedding model parameters

Return Value:

    bool indicating success or failure of the initialization.

--*/
bool 
RAG_initialize(
    _In_ model_params & eparams
    ) 
{
    // Load the DLL first if not already loaded
    if (!LoadLlmInferDll())
    {
        Debug::LogError("Failed to load required DLL for RAG");
        return false;
    }

    if (g_bRagDbInitialized) 
    {
        Debug::Log("RAG database already initialized");
        return true;
    }

    Debug::Log("Initializing RAG database...");
    
    g_pRagDB = nullptr;    

    std::string index_path = "./ragdb/rag_index.bin";
    std::string chunks_path = "./ragdb/chunks.json";
    std::string metadata_path = "./ragdb/metadata.json";
    
    // Check for required files
    if (!std::filesystem::exists(index_path) || !std::filesystem::exists(chunks_path)) 
    {
        Debug::LogError("Required database files not found. Need " + index_path + " and " + chunks_path);
        return false;
    }

    // Load the chunks info
    std::ifstream chunks_json_file(chunks_path);
    json chunks_json;
    try 
    {
        chunks_json = json::parse(chunks_json_file);
    } 
    catch (const std::exception& e) 
    {
        Debug::LogError("Error reading chunk: " + std::string(e.what()));
        return false;
    }

    g_RagEntries.clear();
    for (const auto& item : chunks_json) 
    {
        int id = item["id"];
        std::string source = item["source"];
        std::string text = item["text"];
        rag_entry entry = {id, source, text};
        g_RagEntries.push_back(entry);
    }

    // Get dimension from metadata
    std::string doc_count = "";
    std::ifstream metadata_file(metadata_path);
    if (metadata_file.good()) 
    {
        try 
        {
            json metadata = json::parse(metadata_file);
            if (metadata.contains("dimension")) 
            {
                int meta_dimension = metadata["dimension"];
                if (meta_dimension != eparams.n_dim) 
                {
                    Debug::Log("Using dimension from metadata: " + std::to_string(meta_dimension));
                    eparams.n_dim = meta_dimension;
                }
            }
            doc_count = metadata.contains("document_count") ? std::to_string(metadata["document_count"].get<int>()) : "?";

        } 
        catch (const std::exception& e) 
        {
            Debug::LogError("Error reading metadata: " + std::string(e.what()));
            return false;
        }
        metadata_file.close();
    }

    Debug::Log("Loaded " + std::to_string(g_RagEntries.size()) + " chunks from " + doc_count + " documents");

    // load the database
    try 
    {
        // Load index DB
        Debug::Log("Loading index with dimension " + std::to_string(eparams.n_dim));

        hnswlib::L2Space *rag_space = new hnswlib::L2Space(eparams.n_dim);
        g_pRagDB = new hnswlib::HierarchicalNSW<float>(rag_space, index_path);
        Debug::Log("RAG database loaded from: " + index_path);
                   
    } 
    catch (const std::exception& e) 
    {
        Debug::LogError("Error loading database: " + std::string(e.what()));
        return false;
    }

    size_t max_elements = g_pRagDB->getMaxElements();
    size_t cur_elementCount = g_pRagDB->getCurrentElementCount();
        
    Debug::Log("Loaded vector DB '" + index_path + "' - max_elms(" + 
               std::to_string(max_elements) + ")/cur_elms(" + std::to_string(cur_elementCount) + ")");

    // Initialize embedding model using function pointer
    if (!g_pfnEmbedInitialize(eparams))
    {
        Debug::LogError("Failed to initialize embedding model");
        return false;
    }
    
    // Set flag to indicate readiness for inference
    g_bRagDbInitialized = true;
    return true;
}

/*++

Routine Description:

    Retrieves chunks from the RAG database based on a query.
    Searches for the most relevant chunks to provide as context.

Arguments:

    eparams - Embedding model parameters
    query - The user query to find relevant chunks for
    top_k - Maximum number of chunks to retrieve

Return Value:

    std::vector<rag_entry> - The retrieved chunks

--*/
std::vector<rag_entry> 
retrieve_chunks(
    _In_ const model_params & eparams,
    _In_ const std::string& query, 
    _In_ int top_k
    ) 
{
    std::vector<rag_entry> rag_context;

    if (!g_bRagDbInitialized) 
    {
        Debug::LogError("RAG Database not initialized - please call embed_initialize()");
        return rag_context;
    }

    try 
    {
        // Generate query embedding using embed_encode_single
        std::vector<float> embeddings(eparams.n_dim, 0);
        
        // Use embed_encode_single directly
        if (!g_pfnEmbedEncodeSingle || !g_pfnEmbedEncodeSingle(eparams, query, embeddings)) 
        {
            Debug::LogError("Error during encoding for query: " + query);
            return rag_context;
        }

        // Search for nearest chunks
        try 
        {
            int k = std::min(top_k, static_cast<int>(g_RagEntries.size()));
            std::vector<std::pair<float, hnswlib::labeltype>> results = 
                g_pRagDB->searchKnnCloserFirst(embeddings.data(), k);

            for (auto item: results) 
            {
                int idx = item.second; // chunk index/label
                auto rag_item = g_RagEntries[idx];
                Debug::Log("Retrieved chunk #" + std::to_string(rag_item.id) + 
                         ", distance: " + std::to_string(item.first) + 
                         ", text: " + rag_item.textdata.substr(0, 40) + "...");
                
                if (idx != rag_item.id) 
                {
                    Debug::LogError("Error during retrieval of rag item: id = " + 
                                  std::to_string(idx) + " - expected value = " + 
                                  std::to_string(rag_item.id));
                }
                rag_context.push_back(rag_item);
            }

        } 
        catch (const std::exception& e) 
        {
            Debug::LogError("Search failed: " + std::string(e.what()));
            return rag_context;
        }

    } 
    catch (const std::exception& e) 
    {
        Debug::LogError("Error in retrieve_chunks: " + std::string(e.what()));
    }

    return rag_context;
}

/*++

Routine Description:

    Initializes the embedding model.

Arguments:

    params - Embedding model parameters.

Return Value:

    bool indicating success or failure of the initialization.

--*/
bool 
embed_initialize(
    _In_ model_params& params
    ) 
{
    if (!LoadLlmInferDll())
    {
        Debug::LogError("Failed to load required DLL for embeddings");
        return false;
    }
    
    return g_pfnEmbedInitialize(params);
}

/*++

Routine Description:

    Encodes a batch of chunks using the embedding model.

Arguments:

    params - Embedding model parameters.
    chunks - Vector of chunks to encode.

Return Value:

    bool indicating success or failure of the encoding.

--*/
bool 
embed_encode_batch(
    _In_ const model_params& params,
    _Inout_ std::vector<chunk>& chunks
    ) 
{
    if (!LoadLlmInferDll() || !g_pfnEmbedEncodeBatch)
    {
        Debug::LogError("Failed to load required DLL or embed_encode_batch function");
        return false;
    }
    
    return g_pfnEmbedEncodeBatch(params, chunks);
}

/*++

Routine Description:

    Encodes a single query string using the embedding model.

Arguments:

    params - Embedding model parameters.
    query - The query string to encode.
    embeddings - Output vector to store the embeddings.

Return Value:

    bool indicating success or failure of the encoding.

--*/
bool 
embed_encode_single(
    _In_ const model_params& params,
    _In_ const std::string& query,
    _Out_ std::vector<float>& embeddings
    ) 
{
    if (!LoadLlmInferDll() || !g_pfnEmbedEncodeSingle)
    {
        Debug::LogError("Failed to load required DLL or embed_encode_single function");
        return false;
    }
    
    return g_pfnEmbedEncodeSingle(params, query, embeddings);
}

/*++

Routine Description:

    Terminates the embedding model and releases resources.

Arguments:

    None.

Return Value:

    None.

--*/
void 
embed_terminate(
    void
    ) 
{
    if (g_pfnEmbedTerminate) 
    {
        g_pfnEmbedTerminate();
    }
}
