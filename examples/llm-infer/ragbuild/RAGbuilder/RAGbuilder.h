/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    RAGBuilder.h

Abstract:

    This module defines the RAG database builder class for AIMX.
    Provides functionality to create vector databases from document collections.

Author:

    Rupo Zhang (rizhang) 03/23/2025

--*/

#pragma once

#include "pch.h"
#include "RAGHelper.h"

// Progress callback for reporting database build progress
using RagBuildProgressCallback = std::function<void(const std::string& status, int progressPercent)>;

// Document statistics for metadata
struct DocumentStats {
    std::string filename;
    std::string path;
    std::string fileType; // txt, docx, pdf, etc.
    size_t sizeBytes;
    size_t charCount;
    double extractionTime;
};

// Metadata for the RAG database
struct RagMetadata {
    std::string modelName;
    int dimension;
    int documentCount;
    int docxCount;
    int txtCount;
    int logCount;
    int pdfCount;
    int unsupportedCount;
    int failedCount;
    int chunkCount;
    int chunkSize;
    int totalTokens;
    int indexEfConstruction;
    int indexM;
    int indexEfSearch;
    double embeddingTime;
    std::string documentsPath;
};

// Class for building RAG databases
class RAGBuilder {
public:
    RAGBuilder();
    ~RAGBuilder();

    bool
    Initialize(
        _In_ const model_params& params,
        _In_ const std::string& outputDir,
        _In_ int chunkSize
    );

    // Update method signatures to include the progress callback parameter
    bool
    ProcessDocumentDirectory(
        _In_ const std::string& directoryPath,
        _In_opt_ RagBuildProgressCallback progressCallback = nullptr
    );

    bool
    ProcessDocument(
        _In_ const std::string& filePath,
        _In_opt_ RagBuildProgressCallback progressCallback = nullptr
    );

    bool
    BuildDatabase(
        _In_opt_ RagBuildProgressCallback progressCallback = nullptr
    );

    const RagMetadata&
    GetMetadata() const;

    // Other existing methods...
private:
    // Create a temporary directory for storing files
    std::string CreateTemporaryDirectory();

    // Create chunks from a document using basic or semantic chunking
    std::vector<chunk> ChunkDocument(
        _In_ const rag_entry& document,
        _In_ bool useSemanticChunking = false,
        _In_ float similarityThreshold = 0.7f
    );

    // Generate embeddings for chunks
    bool GenerateEmbeddings(
        _Inout_ std::vector<chunk>& chunks,
        _In_opt_ RagBuildProgressCallback progressCallback = nullptr
    );

    // Save database files
    bool SaveDatabase(
        _In_ const std::vector<chunk>& chunks,
        _In_ hnswlib::HierarchicalNSW<float>* vectorDb
    );

    // Save metadata
    bool SaveMetadata(
        _In_ const std::vector<DocumentStats>& documentStats
    );

    // Extract text from different document types
    std::string ExtractTextFromFile(
        _In_ const std::string& filePath
    );

    // Extract text for text based file eg. txt, log
    std::string ExtractTextBasedFile(
        _In_ const std::string& filePath
    );

    // Extract text for PDF file
    std::string ExtractPDFFile(
        _In_ const std::string& filePath
    );

    // Extract text for Docx file
    std::string ExtractDocxFile(
        _In_ const std::string& filePath
    );

    // Configuration
    model_params m_params;
    std::string m_outputDirectory;
    int m_chunkSize;
    bool m_initialized;

    // Data
    std::vector<rag_entry> m_documents;
    std::vector<DocumentStats> m_documentStats;
    RagMetadata m_metadata;

    // Paths for output files
    std::string m_indexPath;
    std::string m_chunksPath;
    std::string m_metadataPath;
};
