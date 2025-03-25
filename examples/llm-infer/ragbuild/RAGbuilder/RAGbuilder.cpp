/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    RAGBuilder.cpp

Abstract:

    This module implements the RAG database builder.
    Provides functionality to convert document collections into vector databases for RAG.

    This is incorporated from the code:
    %SDXROOT%\xbox\gamecore\so2001\z-slmapp\llm-infer\buildrag\buildrag.cpp

    Note:
    The sentence_embeddings are not yet available in the current version of the code.

Author:

    Rupo Zhang (rizhang) 03/23/2025

Revision history:

    - 03/23/2025:   Initial implementation.
                    Added support for chunking and embedding generation.
                    Added metadata handling and file I/O operations.

To be implemented:
                    Add support for different document types (event logs, Markdown).
                    Add support for basic document formats (PDF, DOCX), they are now placeholders.
                    Optimize the javascript code to the building process for status report.
                    Add database management (such as delete, update, etc.)

--*/

#include "pch.h"
#include "RAGBuilder.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>

namespace fs = std::filesystem;

/*++

Routine Description:

    Constructor for RAGBuilder class. Initializes with default values.

Arguments:

    None.

Return Value:

    None.

--*/
RAGBuilder::RAGBuilder()
    : m_chunkSize(128),
      m_initialized(false),
      m_outputDirectory("."),
      m_indexPath("./rag_index.bin"),
      m_chunksPath("./chunks.json"),
      m_metadataPath("./metadata.json")
{
    // Initialize metadata with zeroes
    memset(&m_metadata, 0, sizeof(RagMetadata));
}

/*++

Routine Description:

    Destructor for RAGBuilder class. Performs cleanup.

Arguments:

    None.

Return Value:

    None.

--*/
RAGBuilder::~RAGBuilder()
{
    // If we've loaded the embedding model, terminate it
    if (m_initialized) 
    {
        // 
        // !BUGBUG!: Rupo Zhang 3/24/2025
        // Cleanup the embedding model
        // for some reason, the cleanup crashed after the db is created. let's debug it later.
        //
        // embed_terminate();
    }
}

/*++

Routine Description:

    Initializes the RAG builder with configuration parameters.

Arguments:

    params - Embedding model parameters.
    outputDir - Directory where RAG database files will be saved.
    chunkSize - Size of text chunks for processing.

Return Value:

    bool indicating success or failure of initialization.

--*/
bool
RAGBuilder::Initialize(
    _In_ const model_params& params,
    _In_ const std::string& outputDir,
    _In_ int chunkSize
)
{
    Debug::Log("Initializing RAG builder with chunk size " + std::to_string(chunkSize));
    
    // Store parameters
    m_params = params;
    m_outputDirectory = outputDir;
    m_chunkSize = chunkSize;
    
    // Update chunk size in params
    m_params.chunk_size = chunkSize;
    
    // Set output file paths
    m_indexPath = m_outputDirectory + "/rag_index.bin";
    m_chunksPath = m_outputDirectory + "/chunks.json";
    m_metadataPath = m_outputDirectory + "/metadata.json";
    
    // Initialize embedding model
    if (!embed_initialize(m_params)) 
    {
        Debug::LogError("Failed to initialize embedding model: " + m_params.model_name);
        return false;
    }
    
    // Clear any existing data
    m_documents.clear();
    m_documentStats.clear();
    
    // Initialize metadata
    m_metadata.modelName = m_params.model_name;
    m_metadata.dimension = m_params.n_dim;
    m_metadata.chunkSize = m_chunkSize;
    
    m_initialized = true;
    return true;
}

/*++

Routine Description:

    Processes a directory of documents for RAG database creation.

Arguments:

    directoryPath - Path to the directory containing documents, or file paths separated by ';'.
    progressCallback - Optional callback for reporting progress.

Return Value:

    bool indicating success or failure of document processing.

--*/
bool
RAGBuilder::ProcessDocumentDirectory(
    _In_ const std::string& directoryPath,
    _In_opt_ RagBuildProgressCallback progressCallback
)
{
    if (!m_initialized) 
    {
        Debug::LogError("RAGBuilder not initialized");
        return false;
    }
    
    Debug::Log("Processing document input: " + directoryPath);
    
    if (progressCallback) 
    {
        progressCallback("Starting document processing...", 0);
    }
    
    bool isMultipleFiles = directoryPath.find(';') != std::string::npos;
    std::string pathToProcess = directoryPath;
    std::string tempDirectory = "";
    
    // Handle multiple file mode
    if (isMultipleFiles) {
        Debug::Log("Multiple file mode detected");
        
        // Create a temporary directory to store the files
        tempDirectory = CreateTemporaryDirectory();
        if (tempDirectory.empty()) {
            Debug::LogError("Failed to create temporary directory for processing files");
            return false;
        }
        
        // Parse the concatenated paths
        std::vector<std::string> filePaths;
        std::string currentPath;
        std::istringstream pathStream(directoryPath);
        
        while (std::getline(pathStream, currentPath, ';')) {
            if (!currentPath.empty()) {
                filePaths.push_back(currentPath);
            }
        }
        
        Debug::Log("Processing " + std::to_string(filePaths.size()) + " individual files");
        
        // Copy each file to the temporary directory
        size_t filesCopied = 0;
        for (const auto& filePath : filePaths) {
            try {
                std::string fileName = fs::path(filePath).filename().string();
                std::string destPath = tempDirectory + "/" + fileName;
                
                // Copy the file
                fs::copy_file(filePath, destPath, fs::copy_options::overwrite_existing);
                Debug::Log("Copied file: " + filePath + " to " + destPath);
                filesCopied++;
            }
            catch (const std::exception& e) {
                Debug::LogError("Failed to copy file: " + filePath + " - " + e.what());
                // Continue with other files
            }
        }
        
        if (filesCopied == 0) {
            Debug::LogError("Failed to copy any files to temporary directory");
            
            // Clean up the temporary directory
            try {
                fs::remove_all(tempDirectory);
            } catch (...) {
                // Ignore cleanup errors
            }
            
            return false;
        }
        
        // Set the path to process to the temporary directory
        pathToProcess = tempDirectory;
    }
    
    // Check if the directory exists
    if (!fs::exists(pathToProcess) || !fs::is_directory(pathToProcess)) 
    {
        Debug::LogError("Directory does not exist: " + pathToProcess);
        return false;
    }
    
    // Count files for progress reporting
    size_t totalFiles = 0;
    for (const auto& entry : fs::directory_iterator(pathToProcess)) 
    {
        if (entry.is_regular_file()) 
        {
            totalFiles++;
        }
    }
    
    if (totalFiles == 0) 
    {
        Debug::LogError("No files found in directory: " + pathToProcess);
        return false;
    }
    
    size_t processedFiles = 0;
    bool success = true;
    
    // Process each file in the directory
    for (const auto& entry : fs::directory_iterator(pathToProcess)) 
    {
        if (entry.is_regular_file()) 
        {
            std::string filePath = entry.path().string();
            
            // Process the file
            try 
            {
                std::string fileExtension = entry.path().extension().string();
                std::transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(),
                              [](unsigned char c) { return std::tolower(c); });

                if (fileExtension == ".docx" ||
                    fileExtension == ".txt" ||
                    fileExtension == ".pdf")
                {
                    if (!ProcessDocument(filePath, nullptr)) 
                    {
                        Debug::LogError("Failed to process file: " + filePath);
                        success = false;
                    }
                } 
                else 
                {
                    Debug::Log("Skipping unsupported file type: " + filePath);
                    m_metadata.unsupportedCount++;
                }
            } 
            catch (const std::exception& e) 
            {
                Debug::LogError("Exception processing file: " + filePath + " - " + e.what());
                m_metadata.failedCount++;
                success = false;
            }
            
            // Report progress at regular intervals
            if (progressCallback && totalFiles > 0) 
            {
                // Calculate progress percentage (0-100)
                int progressPercent = static_cast<int>((processedFiles * 100) / totalFiles);
                progressCallback("Processing file " + std::to_string(processedFiles) + 
                               " of " + std::to_string(totalFiles), progressPercent);
            }
            
            processedFiles++;
        }
    }
    
    // Save document directory in metadata
    m_metadata.documentsPath = isMultipleFiles ? "Multiple files" : pathToProcess;
    
    // Clean up temporary directory if we created one
    if (!tempDirectory.empty()) {
        Debug::Log("Cleaning up temporary directory: " + tempDirectory);
        try {
            fs::remove_all(tempDirectory);
        }
        catch (const std::exception& e) {
            Debug::LogError("Failed to clean up temporary directory: " + std::string(e.what()));
            // Continue anyway, this is just cleanup
        }
    }
    
    if (progressCallback) 
    {
        progressCallback("Document processing complete", 100);
    }
    
    return success;
}

/*++

Routine Description:

    Processes a single document file for RAG database creation.

Arguments:

    filePath - Path to the document file.
    progressCallback - Optional callback for reporting progress.

Return Value:

    bool indicating success or failure of document processing.

--*/
bool
RAGBuilder::ProcessDocument(
    _In_ const std::string& filePath,
    _In_opt_ RagBuildProgressCallback progressCallback
)
{
    if (!m_initialized) 
    {
        Debug::LogError("RAGBuilder not initialized");
        return false;
    }
    
    Debug::Log("Processing document: " + filePath);
    
    if (progressCallback) 
    {
        progressCallback("Processing document: " + fs::path(filePath).filename().string(), 0);
    }
    
    try 
    {
        // Check if the file exists
        if (!fs::exists(filePath) || !fs::is_regular_file(filePath)) 
        {
            Debug::LogError("File does not exist: " + filePath);
            return false;
        }
        
        // Get file info
        std::string fileBaseName = fs::path(filePath).filename().string();
        std::string fileExtension = fs::path(filePath).extension().string();
        std::transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(),
                      [](unsigned char c) { return std::tolower(c); });
        
        // Track timing
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Extract text from file
        std::string textContent = ExtractTextFromFile(filePath);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> extractionTime = endTime - startTime;
        
        if (textContent.empty()) 
        {
            Debug::LogError("Failed to extract text from: " + filePath);
            m_metadata.failedCount++;
            return false;
        }
        
        if (progressCallback) 
        {
            progressCallback("Text extracted, creating document entry", 50);
        }
        
        // Create document entry
        rag_entry document;
        document.id = (int)m_documents.size();
        document.filename = fileBaseName;
        document.textdata = textContent;
        
        // Add to document collection
        m_documents.push_back(document);
        
        // Create document stats
        DocumentStats stats;
        stats.filename = fileBaseName;
        stats.path = filePath;
        stats.fileType = fileExtension.substr(1); // Remove the dot
        stats.sizeBytes = fs::file_size(filePath);
        stats.charCount = textContent.size();
        stats.extractionTime = extractionTime.count();
        
        // Add to stats collection
        m_documentStats.push_back(stats);
        
        // Update metadata
        m_metadata.documentCount++;
        
        if (fileExtension == ".docx") 
        {
            m_metadata.docxCount++;
        } 
        else if (fileExtension == ".txt") 
        {
            m_metadata.txtCount++;
        } 
        else if (fileExtension == ".pdf") 
        {
            m_metadata.pdfCount++;
        } 
        else 
        {
            m_metadata.unsupportedCount++;
        }
        
        if (progressCallback) 
        {
            progressCallback("Document processed successfully", 100);
        }
        
        return true;
    } 
    catch (const std::exception& e) 
    {
        Debug::LogError("Exception processing document: " + std::string(e.what()));
        m_metadata.failedCount++;
        return false;
    }
}

/*++

Routine Description:

    Extracts text content from a document file.
    Currently supports basic text extraction, can be extended for various formats.

Arguments:

    filePath - Path to the document file.

Return Value:

    std::string containing the extracted text content.

--*/
std::string
RAGBuilder::ExtractTextFromFile(
    _In_ const std::string& filePath
)
{
    std::string fileExtension = fs::path(filePath).extension().string();
    std::transform(fileExtension.begin(), fileExtension.end(), fileExtension.begin(),
                  [](unsigned char c) { return std::tolower(c); });

    //
    // !FIXME! - Rupo Zhang 3/24/2025 
    // For now, we need to incorporate proper implementation for DOCX and PDF extraction.
    // it is all place holder except the text (.txt) file.
    //

    if (fileExtension == ".txt") 
    {
        // Read text file directly
        std::ifstream file(filePath);
        if (!file.is_open()) 
        {
            Debug::LogError("Failed to open text file: " + filePath);
            return "";
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    } 
    else if (fileExtension == ".docx") 
    {
        // Use external tool for DOCX files
        // This would need to be implemented separately
        Debug::Log("DOCX extraction requires external document handling tools");
        // For the prototype, we'll pretend it worked but return placeholder text
        return "This is placeholder text for DOCX file: " + fs::path(filePath).filename().string();
    } 
    else if (fileExtension == ".pdf") 
    {
        // Use external tool for PDF files
        Debug::Log("PDF extraction requires external document handling tools");
        // For the prototype, we'll pretend it worked but return placeholder text
        return "This is placeholder text for PDF file: " + fs::path(filePath).filename().string();
    }
    
    // Unsupported file type
    Debug::LogError("Unsupported file type for text extraction: " + fileExtension);
    return "";
}

/*++

Routine Description:

    Creates chunks from a document for processing and embedding.

Arguments:

    document - The document to chunk.

Return Value:

    std::vector<chunk> containing the document chunks.

--*/
std::vector<chunk>
RAGBuilder::ChunkDocument(
    _In_ const rag_entry& document
)
{
    std::vector<chunk> chunks;
    
    // Skip empty documents
    if (document.textdata.empty()) 
    {
        Debug::Log("Skipping empty document: " + document.filename);
        return chunks;
    }
    
    const std::string& texts = document.textdata;
    
    // Create a new chunk
    chunk currentChunk = {document.filename, "", {}, {}};
    
    // Define sentence separators
    const std::string separators = ".!?";
    
    size_t start = 0;
    size_t end = 0;
    
    // Split by sentences
    while ((end = texts.find_first_of(separators, start)) != std::string::npos) 
    {
        // Extract the sentence
        std::string textData = texts.substr(start, end - start + 1);
        
        // Check if adding this would exceed chunk size
        if (currentChunk.textdata.size() + textData.size() > (size_t)m_chunkSize) 
        {
            // Current chunk is large enough, add it to the list
            if (!currentChunk.textdata.empty()) 
            {
                chunks.push_back(currentChunk);
                // Create a new chunk with this sentence
                currentChunk.textdata = textData;
            } 
            else 
            {
                // This single sentence is larger than chunk size, add it anyway
                currentChunk.textdata = textData;
                chunks.push_back(currentChunk);
                currentChunk.textdata = "";
            }
        } 
        else 
        {
            // Add sentence to current chunk
            currentChunk.textdata += textData;
        }
        
        // Move to next position after the separator
        start = end + 1;
    }
    
    // Handle any remaining text
    if (start < texts.size()) 
    {
        std::string remainingText = texts.substr(start);
        
        if (currentChunk.textdata.size() + remainingText.size() > (size_t)m_chunkSize) 
        {
            // Current chunk is getting too large
            if (!currentChunk.textdata.empty()) 
            {
                chunks.push_back(currentChunk);
                currentChunk.textdata = "";
            }
            
            // Process remaining text in chunks
            size_t pos = 0;
            while (pos < remainingText.size()) 
            {
                size_t length = std::min((size_t)m_chunkSize, remainingText.size() - pos);
                currentChunk.textdata = remainingText.substr(pos, length);
                chunks.push_back(currentChunk);
                currentChunk.textdata = "";
                pos += length;
            }
        } 
        else 
        {
            // Add remaining text to current chunk
            currentChunk.textdata += remainingText;
        }
    }
    
    // Add the last chunk if it's not empty
    if (!currentChunk.textdata.empty()) 
    {
        chunks.push_back(currentChunk);
    }
    
    return chunks;
}

/*++

Routine Description:

    Generates embeddings for document chunks using the embedding model.

Arguments:

    chunks - Vector of chunks to generate embeddings for.
    progressCallback - Optional callback for reporting progress.

Return Value:

    bool indicating success or failure of embedding generation.

--*/
bool
RAGBuilder::GenerateEmbeddings(
    _Inout_ std::vector<chunk>& chunks,
    _In_opt_ RagBuildProgressCallback progressCallback
)
{
    if (chunks.empty()) 
    {
        Debug::LogError("No chunks to generate embeddings for");
        return false;
    }
    
    Debug::Log("Generating embeddings for " + std::to_string(chunks.size()) + " chunks");
    
    if (progressCallback) 
    {
        progressCallback("Starting embeddings generation", 0);
    }
    
    // Track timing
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Use the embedding model to generate embeddings
    bool success = embed_encode_batch(m_params, chunks);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> embeddingTime = endTime - startTime;
    
    // Store embedding time in metadata
    m_metadata.embeddingTime = embeddingTime.count();
    
    // Process chunks in batches with progress reporting
    for (size_t i = 0; i < chunks.size(); i++) 
    {  
        // Update progress every few chunks
        if (progressCallback && i % std::max<size_t>(1, chunks.size() / 20) == 0) 
        {
            int percent = static_cast<int>((i * 100) / chunks.size());
            progressCallback("Generated embeddings for " + std::to_string(i) + 
                           " of " + std::to_string(chunks.size()) + " chunks", percent);
        }
    }
    
    if (!success) 
    {
        Debug::LogError("Failed to generate embeddings for chunks");
        return false;
    }
    
    if (progressCallback) 
    {
        progressCallback("Embeddings generation complete", 100);
    }
    
    Debug::Log("Embeddings generated in " + std::to_string(embeddingTime.count()) + " seconds");
    return true;
}

/*++

Routine Description:

    Builds a RAG database from processed documents.

Arguments:

    progressCallback - Optional callback for reporting progress.

Return Value:

    bool indicating success or failure of database building.

--*/
bool
RAGBuilder::BuildDatabase(
    _In_opt_ RagBuildProgressCallback progressCallback
)
{
    if (!m_initialized) 
    {
        Debug::LogError("RAGBuilder not initialized");
        return false;
    }
    
    if (m_documents.empty()) 
    {
        Debug::LogError("No documents to build database from");
        return false;
    }
    
    Debug::Log("Building RAG database from " + std::to_string(m_documents.size()) + " documents");
    
    if (progressCallback) 
    {
        progressCallback("Starting database build", 0);
    }
    
    Debug::Log("Chunking documents...");
    
    // Collection of all chunks across all documents
    std::vector<chunk> allChunks;
    
    // Process each document into chunks
    for (size_t i = 0; i < m_documents.size(); i++) 
    {
        std::vector<chunk> documentChunks = ChunkDocument(m_documents[i]);
        Debug::Log("Document " + m_documents[i].filename + ": " + 
                  std::to_string(documentChunks.size()) + " chunks");
        
        // Add to collection
        allChunks.insert(allChunks.end(), documentChunks.begin(), documentChunks.end());
    }
    
    // Update metadata
    m_metadata.chunkCount = (int)allChunks.size();
    
    if (allChunks.empty()) 
    {
        Debug::LogError("No chunks generated from documents");
        return false;
    }
    
    // Chunking phase
    if (progressCallback) 
    {
        progressCallback("Chunking documents", 10);
    }
    
    Debug::Log("Generating embeddings...");
    
    // Embedding generation phase
    if (progressCallback) 
    {
        progressCallback("Generating embeddings", 30);
    }
    
    // Generate embeddings
    if (!GenerateEmbeddings(allChunks, 
        [progressCallback](const std::string& status, int percent) {
            // Map embedding progress to 30-70% range
            if (progressCallback) {
                int adjustedPercent = 30 + (percent * 40) / 100;
                progressCallback(status, adjustedPercent);
            }
        })) 
    {
        Debug::LogError("Failed to generate embeddings");
        return false;
    }
    
    // Count tokens
    for (const auto& chunk : allChunks) 
    {
        m_metadata.totalTokens += (int)chunk.tokens.size();
    }
    
    Debug::Log("Building vector database...");
    
    // Database construction phase
    if (progressCallback) 
    {
        progressCallback("Building vector database", 70);
    }
    
    // Create HNSW index
    
    // HNSW parameters
    int maxElements = std::max(10000, (int)(allChunks.size() * 1.5)); // Some buffer room
    int M = 16;               // Links per element
    int efConstruction = 200; // Build time vs accuracy tradeoff
    int efSearch = 100;       // Search accuracy parameter
    
    // Update metadata
    m_metadata.indexM = M;
    m_metadata.indexEfConstruction = efConstruction;
    m_metadata.indexEfSearch = efSearch;
    
    Debug::Log("Creating HNSW index with M=" + std::to_string(M) + 
              ", efConstruction=" + std::to_string(efConstruction));
    
    // Initialize index
    hnswlib::L2Space space(m_params.n_dim);
    hnswlib::HierarchicalNSW<float>* hnsw = 
        new hnswlib::HierarchicalNSW<float>(&space, maxElements, M, efConstruction);
    hnsw->setEf(efSearch);
    
    // Add all vectors to the index
    for (size_t i = 0; i < allChunks.size(); i++) 
    {
        // Add point with ID equal to its position
        hnsw->addPoint(allChunks[i].embeddings.data(), (hnswlib::labeltype)i);
    }
    
    Debug::Log("Saving database files...");
    
    // Saving phase
    if (progressCallback) 
    {
        progressCallback("Saving database files", 90);
    }
    
    // Save the database
    bool success = SaveDatabase(allChunks, hnsw);
    
    // Clean up
    delete hnsw;
    
    if (success) 
    {
        Debug::Log("RAG database built successfully with " + 
                  std::to_string(allChunks.size()) + " chunks");
        if (progressCallback) 
        {
            progressCallback("Database build complete", 100);
        }
        return true;
    } 
    else 
    {
        Debug::LogError("Failed to save RAG database");
        return false;
    }
}

/*++

Routine Description:

    Saves the RAG database files (vector index and chunk data).

Arguments:

    chunks - The document chunks with embeddings.
    vectorDb - The HNSW vector database.

Return Value:

    bool indicating success or failure of database saving.

--*/
bool
RAGBuilder::SaveDatabase(
    _In_ const std::vector<chunk>& chunks,
    _In_ hnswlib::HierarchicalNSW<float>* vectorDb
)
{
    Debug::Log("Saving RAG database to: " + m_indexPath);
    
    try 
    {
        // Create output directory if it doesn't exist
        fs::create_directories(m_outputDirectory);
        
        // Save the HNSW index
        vectorDb->saveIndex(m_indexPath);
        
        Debug::Log("Vector database saved: max_elements=" + 
                  std::to_string(vectorDb->getMaxElements()) + 
                  ", current_elements=" + 
                  std::to_string(vectorDb->getCurrentElementCount()));
        
        // Create chunks JSON
        nlohmann::json chunksJson = nlohmann::json::array();
        
        for (size_t i = 0; i < chunks.size(); i++) 
        {
            nlohmann::json chunkObj;
            chunkObj["id"] = i;
            chunkObj["source"] = chunks[i].filename;
            chunkObj["text"] = chunks[i].textdata;
            chunkObj["tokens"] = chunks[i].tokens.size();
            
            chunksJson.push_back(chunkObj);
        }
        
        // Save chunks JSON
        std::ofstream chunksFile(m_chunksPath);
        if (!chunksFile.is_open()) 
        {
            Debug::LogError("Failed to open chunks file for writing: " + m_chunksPath);
            return false;
        }
        
        chunksFile << chunksJson.dump(2); // Pretty print with 2 spaces
        chunksFile.close();
        
        Debug::Log("Chunks data saved: " + std::to_string(chunks.size()) + " chunks");
        
        // Save metadata
        if (!SaveMetadata(m_documentStats)) 
        {
            Debug::LogError("Failed to save metadata");
            return false;
        }
        
        return true;
    } 
    catch (const std::exception& e) 
    {
        Debug::LogError("Exception saving database: " + std::string(e.what()));
        return false;
    }
}

/*++

Routine Description:

    Saves metadata about the RAG database.

Arguments:

    documentStats - Statistics about the processed documents.

Return Value:

    bool indicating success or failure of metadata saving.

--*/
bool
RAGBuilder::SaveMetadata(
    _In_ const std::vector<DocumentStats>& documentStats
)
{
    Debug::Log("Saving RAG metadata to: " + m_metadataPath);
    
    try 
    {
        // Get current time
        auto currentTime = std::chrono::system_clock::now();
        std::time_t currentTimeT = std::chrono::system_clock::to_time_t(currentTime);
        char timeBuffer[100];
        std::strftime(timeBuffer, sizeof(timeBuffer), "%Y-%m-%d %H:%M:%S", std::localtime(&currentTimeT));
        
        // Create metadata JSON
        nlohmann::json metadata;
        
        // Basic info
        metadata["created_at"] = timeBuffer;
        metadata["model"] = m_metadata.modelName;
        metadata["documents_path"] = m_metadata.documentsPath;
        metadata["dimension"] = m_metadata.dimension;
        metadata["document_count"] = m_metadata.documentCount;
        
        // Document types
        nlohmann::json documentTypes;
        documentTypes["docx"] = m_metadata.docxCount;
        documentTypes["txt"] = m_metadata.txtCount;
        documentTypes["pdf"] = m_metadata.pdfCount;
        documentTypes["unsupported"] = m_metadata.unsupportedCount;
        documentTypes["failed"] = m_metadata.failedCount;
        metadata["document_types"] = documentTypes;
        
        // Chunk info
        metadata["chunk_count"] = m_metadata.chunkCount;
        metadata["chunk_size"] = m_metadata.chunkSize;
        metadata["total_tokens"] = m_metadata.totalTokens;
        
        // Index params
        nlohmann::json indexParams;
        indexParams["ef_construction"] = m_metadata.indexEfConstruction;
        indexParams["M"] = m_metadata.indexM;
        indexParams["ef_search"] = m_metadata.indexEfSearch;
        metadata["index_params"] = indexParams;
        
        // Performance
        metadata["embedding_time"] = m_metadata.embeddingTime;
        metadata["verbose"] = m_params.verbose;
        
        // Document details
        nlohmann::json documents = nlohmann::json::array();
        
        for (const auto& docStat : documentStats) 
        {
            nlohmann::json doc;
            doc["filename"] = docStat.filename;
            doc["path"] = docStat.path;
            doc["size_bytes"] = docStat.sizeBytes;
            doc["extraction_time"] = docStat.extractionTime;
            doc["char_count"] = docStat.charCount;
            doc["type"] = docStat.fileType;
            
            documents.push_back(doc);
        }
        
        metadata["documents"] = documents;
        
        // Write to file
        std::ofstream metadataFile(m_metadataPath);
        if (!metadataFile.is_open()) 
        {
            Debug::LogError("Failed to open metadata file for writing: " + m_metadataPath);
            return false;
        }
        
        metadataFile << metadata.dump(2); // Pretty print with 2 spaces
        metadataFile.close();
        
        Debug::Log("Metadata saved successfully");
        return true;
    } 
    catch (const std::exception& e) 
    {
        Debug::LogError("Exception saving metadata: " + std::string(e.what()));
        return false;
    }
}

/*++

Routine Description:

    Gets the metadata about the built RAG database.

Arguments:

    None.

Return Value:

    const RagMetadata& - Reference to the metadata structure.

--*/
const RagMetadata&
RAGBuilder::GetMetadata() const
{
    return m_metadata;
}

/*++

Routine Description:

    Creates a temporary directory for storing selected files.

Arguments:

    None.

Return Value:

    std::string containing the path to the temporary directory.

--*/
std::string
RAGBuilder::CreateTemporaryDirectory() 
{
    // Create a unique temporary directory path
    wchar_t tempPath[MAX_PATH];
    DWORD result = GetTempPathW(MAX_PATH, tempPath);
    if (result == 0 || result > MAX_PATH) {
        Debug::LogError("Failed to get temporary path");
        return "";
    }
    
    // Create a unique folder name with timestamp
    std::wstring uniqueName = L"aimx_rag_" + std::to_wstring(std::chrono::system_clock::now().time_since_epoch().count());
    std::wstring tempDir = std::wstring(tempPath) + uniqueName;
    
    // Create the directory
    if (!CreateDirectoryW(tempDir.c_str(), NULL) && GetLastError() != ERROR_ALREADY_EXISTS) {
        Debug::LogError("Failed to create temporary directory");
        return "";
    }
    
    Debug::Log("Created temporary directory: " + WideToUtf8(tempDir));
    return WideToUtf8(tempDir);
}
