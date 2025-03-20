#pragma warning (disable:4100) // unref formal param
#pragma warning (disable:4127) // cond expression is constant
#pragma warning (disable:4242) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4244) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4245) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4267) // conversion from 'int64_t' to 'int'
#pragma warning (disable:4505) // unref function with internal linkage removed

#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <chrono>

using namespace std;
namespace fs = filesystem;

#include "dtt.h"
#include "llm-infer.h"

#include "json.hpp"
using json = nlohmann::json;

// Configuration paths
std::unordered_map<std::string, std::string> DB_CONFIG = {
    {"index_path", "./rag_index.bin"},
    {"chunks_path", "./chunks.json"},
    {"metadata_path", "./metadata.json"}
};

struct metadata_stats {
    string model;
    int dimension;
    int document_count;
    int document_docx_count;
    int document_txt_count;
    int document_pdf_count;
    int document_unsupported_count;
    int document_failed_count;
    int chunk_count;
    int chunk_size;
    int total_tokens;
    int index_ef_construction;
    int index_M;
    int index_ef_search;
    int verbose;
    double embedding_time;
    string document_base_path;
};

struct document_stats {
    string filename;
    string path;
    int size_bytes;
    double extraction_time;
    uint64_t char_count;
    string type;
};

static vector<chunk> chunk_file_by_separators(
    const rag_entry & rag_entry, 
    int chunk_size = 128) {

    vector<chunk> chunks;

    const std::string filename = rag_entry.filename;
    const std::string & texts = rag_entry.textdata;
    size_t text_size = texts.size();
    if (text_size == 0) {
        return chunks;
    }

    chunk current_chunk = {filename, "", {0}, {0.0}};
    size_t start = 0;
    size_t end = 0;

    const string separators = ".!?";
    while ((end = texts.find_first_of(separators, start)) != string::npos) {
        // Extract the sentence into the current chunk
        string textdata = texts.substr(start, end - start + 1);

        if (current_chunk.textdata.size() > chunk_size) {
            // Flush current chunk as it is large enough
            chunks.push_back(current_chunk);
            // Reset for next chunk
            current_chunk.textdata = textdata;
        } else {
            // Keep appending the data to the current chunk
            current_chunk.textdata += textdata;
        }

        // Advance the cursor past the current separator
        start = end + 1;
    }

    // Check for remaining text left over without any separator
    if (start < texts.size()) {
        std::string remaining_text = texts.substr(start);
        if (remaining_text.size() > chunk_size) {
            size_t pos = 0;
            while (pos < remaining_text.size()) {
                size_t len = min(chunk_size, (int)(remaining_text.size() - pos));
                current_chunk.textdata = remaining_text.substr(pos, len);
                chunks.push_back(current_chunk);
                pos += len;
            }
        } else {
            // The remaining is less than chunk size so consume it all
            current_chunk.textdata = remaining_text;
            chunks.push_back(current_chunk);
        }
    }

    return chunks;
}

static vector<chunk> chunk_file_by_sentences(
    const rag_entry & rag_entry, 
    int chunk_size = 1000) {

    vector<chunk> chunks;

    const std::string filename = rag_entry.filename;
    const std::string & texts = rag_entry.textdata;
    size_t text_size = texts.size();
    if (text_size == 0) {
        return chunks;
    }

    vector<std::string> sentences;
    string current;

    const string separators = ".!?";
    size_t start = 0;
    size_t end = 0;

    while ((end = texts.find_first_of(separators, start)) != string::npos) {
        string textdata = texts.substr(start, end - start + 1);
        sentences.push_back(textdata);
        start = end + 1;
    }

    if (!current.empty()) {
        // Save any leftover parts as the last sentence
        sentences.push_back(current);
    }

    if (sentences.empty()) {
        // Return if there is nothing to work on
        return chunks;
    }

#if 0 // TBD - sentence_embeddings are not yet available
    vector<string> chunks;
    vector<string> current_chunk;
    int current_length = 0;

    // Create chunks based on semantic similarity
    for (size_t i = 0; i < sentences.size(); ++i) {
        int sentence_len = sentences[i].size();

        if (current_length + sentence_len > chunk_size && !current_chunk.empty()) {
            chunks.push_back(std::accumulate(current_chunk.begin(), current_chunk.end(), std::string("")));
            current_chunk = { sentences[i] };
            current_length = sentence_len;
        } else {
            if (i > 0 && !current_chunk.empty()) {
                // Calculate cosine similarity
                float dot_product = std::inner_product(sentence_embeddings[i].begin(), sentence_embeddings[i].end(), sentence_embeddings[i - 1].begin(), 0.0f);
                float norm_a = std::sqrt(std::inner_product(sentence_embeddings[i].begin(), sentence_embeddings[i].end(), sentence_embeddings[i].begin(), 0.0f));
                float norm_b = std::sqrt(std::inner_product(sentence_embeddings[i - 1].begin(), sentence_embeddings[i - 1].end(), sentence_embeddings[i - 1].begin(), 0.0f));
                float sim = dot_product / (norm_a * norm_b + 1e-8);

                if (sim < similarity_threshold && !current_chunk.empty()) {
                    chunks.push_back(std::accumulate(current_chunk.begin(), current_chunk.end(), std::string("")));
                    current_chunk = { sentences[i] };
                    current_length = sentence_len;
                } else {
                    current_chunk.push_back(sentences[i]);
                    current_length += sentence_len;
                }
            } else {
                current_chunk.push_back(sentences[i]);
                current_length += sentence_len;
            }
        }
    }

    if (!current_chunk.empty()) {
        chunks.push_back(std::accumulate(current_chunk.begin(), current_chunk.end(), std::string("")));
    }
#endif

    return chunks;
}

vector<chunk> semantic_chunking(
    const model_params & params,
    const vector<rag_entry> & rag_docs
    ) {

    vector<chunk> rag_chunks;

    for (auto item : rag_docs) {
        vector<chunk> item_chunks = chunk_file_by_separators(item, params.chunk_size);
        cout << "    processing " << item.filename << ": " << item_chunks.size() << " chunks" << endl;
        rag_chunks.insert(rag_chunks.end(), item_chunks.begin(), item_chunks.end());
    }

    return rag_chunks;
}

static void generate_metadata_json(
    const metadata_stats & meta_stats,
    const vector<document_stats> & alldocs_stats,
    const string documents_path) {

    auto currentTime = std::chrono::system_clock::now();
    std::time_t currentTimeT = std::chrono::system_clock::to_time_t(currentTime);

    json metadata;

    // Populate the root object with the specified fields
    metadata["created_at"] = ctime(&currentTimeT);
    metadata["model"] = meta_stats.model;
    metadata["documents_path"] = documents_path;
    metadata["dimension"] = meta_stats.dimension;
    metadata["document_count"] = meta_stats.document_count;

    // Create the document_types object
    json document_types;
    document_types["docx"] = meta_stats.document_docx_count;
    document_types["txt"] = meta_stats.document_txt_count;
    document_types["pdf"] = meta_stats.document_pdf_count;
    document_types["unsupported"] = meta_stats.document_unsupported_count;
    document_types["failed"] = meta_stats.document_failed_count;
    metadata["document_types"] = document_types;

    metadata["chunk_count"] = meta_stats.chunk_count;
    metadata["chunk_size"] = meta_stats.chunk_size;
    metadata["total_tokens"] = meta_stats.total_tokens;

    // Create the index_params object
    json index_params;
    index_params["ef_construction"] = meta_stats.index_ef_construction;
    index_params["M"] = meta_stats.index_M;
    index_params["ef_search"] = meta_stats.index_ef_search;
    metadata["index_params"] = index_params;

    metadata["embedding_time"] = meta_stats.embedding_time;
    metadata["verbose"] = meta_stats.verbose;

    // Create the documents array
    json documents = json::array();

    for (auto item : alldocs_stats) {
        json doc;
        doc["filename"] = item.filename;
        doc["path"] = item.path;
        doc["size_bytes"] = 0;
        doc["extraction_time"] = item.extraction_time;
        doc["char_count"] = item.char_count;
        doc["type"] = item.type;
        documents.push_back(doc);
    }

    metadata["documents"] = documents;

    // Persist the job metadata to disk
    std::ofstream metadata_json_file(DB_CONFIG["metadata_path"]);
    if (!metadata_json_file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
    } else {
        // Use 2 spaces indentation
        metadata_json_file << metadata.dump(2);
        metadata_json_file.close();
    }
}

void test_run(
    const model_params & eparams) {
    // Reopen the database and iterate through every chunk and look
    // for the closest chunk based on the text from the same chunk

    string hnsw_path = DB_CONFIG["index_path"];
    hnswlib::L2Space ragspace(eparams.n_dim);
    hnswlib::HierarchicalNSW<float> *rag_db = new hnswlib::HierarchicalNSW<float>(&ragspace, hnsw_path);
    size_t max_elements = rag_db->getMaxElements();
    size_t cur_elementCount = rag_db->getCurrentElementCount();

    printf("[%s]: load vector DB '%s' - max_elms(%zd)/cur_elms(%zd)\n", __func__, hnsw_path.c_str(),
        max_elements, cur_elementCount);

    ifstream test_json_file(DB_CONFIG["chunks_path"]);
    json chunks_json;
    try {
        chunks_json = json::parse(test_json_file);
    } catch (const exception& e) {
        cerr << "Error reading chunk: " << e.what() << endl;
        return;
    }
    vector<rag_entry> rag_entries;
    for (const auto& item : chunks_json) {
        int id = item["id"];
        string source = item["source"];
        string text = item["text"];
        rag_entry entry = {id, source, text};
        rag_entries.push_back(entry);
    }

    for (auto rag_i : rag_entries) {
        // Generate query embedding
        vector<float> embeddings(eparams.n_dim, 0);
        printf("---- Searching for [%3d][%.60s]\n", rag_i.id, rag_i.textdata.c_str());
        if (!embed_encode_single(eparams, rag_i.textdata, embeddings)) {
            printf("%s: error during encoding for query '%s'\n", __func__, rag_i.textdata.c_str());
        }

        // Search for nearest chunks
        try {
            std::vector<std::pair<float, hnswlib::labeltype>> results = rag_db->searchKnnCloserFirst(embeddings.data(), 3);

            for (auto item: results) {
                int idx = /* chunk index/label */ item.second;

                // auto rag_ii = rag_entries[idx];
                auto rag_ii = rag_entries.at(idx);
                printf("     [%4d]. Distance: %5.2f - [%d]-<%.50s>\n", rag_ii.id, item.first, idx, rag_ii.textdata.c_str());
                if (idx != rag_ii.id) {
                    printf("%s: Error during retrieval of rag item: id = %4d - expected value = %4d\n", 
                        __func__, idx, rag_ii.id);
                }
            }
        } catch (const exception& e) {
            cerr << "Search failed: " << e.what() << endl;
        }
    }
    delete rag_db;
}

int main(int argc, char *argv[]) {
    metadata_stats meta_stats;
    vector<document_stats> alldocs_stat;
    memset(&meta_stats, 0, sizeof(metadata_stats));

    if (argc > 1) {
        // Command line argument wins
    } else {
        // Pick a default directory
    }

    // Initialize sentencepiece config defaults
    model_params eparams;
    eparams.model_name = "./models/all-minilm-l6-v2_f32.gguf";
    eparams.n_ctx = 512; // this is highly dependent on the trained model

    // Grab config from metadata file if it exists to override current defaults
    string documents_path = "docs";
    ifstream metadata_json_file(DB_CONFIG["metadata_path"]);
    try {
        if (metadata_json_file.is_open()) {
            json config_json = json::parse(metadata_json_file);
            if (config_json.contains("chunk_size")) {
                int32_t config_chunk_size = config_json["chunk_size"];
                if (eparams.chunk_size > config_chunk_size) {
                    // Override defaults if chunk size from metadata.json is smaller
                    cout << "- Switching to chunk size from metadata json: " << config_chunk_size << endl;
                    eparams.chunk_size = config_chunk_size;
                }
            }
            if (config_json.contains("model")) {
                string config_model = config_json["model"];
                if (eparams.model_name != config_model) {
                    // Override defaults if model name from metadata.json is different
                    cout << "- Switching to embedding model from metadata json: " << config_model << endl;
                    eparams.model_name = config_model;
                }
            }
            if (config_json.contains("documents_path")) {
                string config_documents_path = config_json["documents_path"];
                if (argc == 1) {
                    // If there is no arg then use the path from the metadata
                    cout << "- Using docs directory path from metadata json: [\"" << config_documents_path << "\"]" << endl;
                    documents_path = config_documents_path;
                } else {
                    // User the command line arg if specified
                    cout << "- Using docs directory path from command line args: [\"" << config_documents_path << "\"]" << endl;
                    documents_path = argv[1];
                }
            }
            if (config_json.contains("verbose")) {
                int config_verbose = config_json["verbose"];
                if (eparams.verbose != config_verbose) {
                    // Override defaults if model name from metadata.json is different
                    cout << "- Switching verbose mode as specified in metadata json " << config_verbose << endl;
                    eparams.verbose = config_verbose;
                }
            }
        }
        metadata_json_file.close();

    } catch (const exception& /* e */) {
        cerr << "Skipping out early reading config: " << DB_CONFIG["metadata_path"] << endl;
    }

    vector<rag_entry> rag_docs;
    rag_entry rag_current;
    try {
        // Iterate over the whole directory and process all *.docx files
        for (const auto& entry : fs::directory_iterator(documents_path.c_str())) {
            if (entry.is_regular_file() && entry.path().extension() == ".docx") {
                rag_current.filename = "" + entry.path().stem().string() + ".docx";
                auto start = std::chrono::high_resolution_clock::now();
                string texts = dtt(entry.path());
                auto end = std::chrono::high_resolution_clock::now();
                rag_current.textdata = texts;

                rag_docs.push_back(rag_current);

                std::chrono::duration<double> duration = end - start;
                meta_stats.document_docx_count++;
                document_stats doc_stat = {
                    rag_current.filename,
                    entry.path().string(),
                    0,
                    (double) duration.count(),
                    texts.size(),
                    "docx"
                };
                alldocs_stat.push_back(doc_stat);
            }
        }
    } catch (const fs::filesystem_error& e) {
        cerr << "Filesystem error: " << e.what() << endl;
        meta_stats.document_failed_count++;
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        meta_stats.document_failed_count++;
    }

    // Save all texts for verification
    // flushToTextFile("./summary.txt", full_rag_data);

    vector<chunk> rag_chunks = semantic_chunking(eparams, rag_docs);
    printf("%s: total chunks generated - %zd - chunk size %d\n", __func__, rag_chunks.size(), eparams.chunk_size);

    // Initialize sentencepiece model for embeddings creation
    if (!embed_initialize(eparams)) {
        printf("Initialiazation of embedding model '%s' failed\n", eparams.model_name.c_str());
        return -1;
    }

    // Create embeddings and collect timings
    auto embed_start = std::chrono::high_resolution_clock::now();
    if (!embed_encode_batch(eparams, rag_chunks)) {
        printf("%s: failed to batch encoding during RAG index creation\n", __func__);
        return -1;
    }
    auto embed_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> embed_duration = embed_end - embed_start;
    cout << "Emdedding time taken: " << embed_duration.count() << " seconds" << std::endl;

    int max_elements = 10000;   // Maximum number of elements, should be known beforehand

    // if dataset is big and memory is a concern then NHSW need to be chosen. 
    // As well as in NMSLIB, we chose the M parameter. The 4 <= M <= 64 is 
    // the number of links per vector, higher is more accurate but uses more RAM. 
    // The memory usage is (d * 4 + M * 2 * 4) bytes per vector.
    
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_search = 100;

    // Initializing index
    hnswlib::L2Space rag_space(eparams.n_dim);
    hnswlib::HierarchicalNSW<float>* rag_hnsw = new hnswlib::HierarchicalNSW<float>(&rag_space, max_elements, M, ef_construction);
    rag_hnsw->setEf(ef_search);

    // Create the root JSON array
    json rag_json_root = json::array();

    // Create RAG database
    for (size_t i = 0; i < rag_chunks.size(); i++) {
        json rag_json_obj;
        rag_json_obj["id"] = i;
        rag_json_obj["source"] = rag_chunks[i].filename;
        rag_json_obj["text"] = rag_chunks[i].textdata;
        rag_json_obj["tokens"] = rag_chunks[i].tokens.size();
        // Add the objects to the root array
        rag_json_root.push_back(rag_json_obj);

        meta_stats.total_tokens += rag_chunks[i].tokens.size();

        // Insert the chunk embeddings into the database
        rag_hnsw->addPoint(rag_chunks[i].embeddings.data(), i);
    }

    printf("[%s]: persist vector DB '%s' dimension: %d\n", __func__, DB_CONFIG["index_path"].c_str(), eparams.n_dim);
    printf("[%s]: vector DB max_elements-[%zd] / cur_elements-[%zd]\n", __func__,
        rag_hnsw->getMaxElements(), rag_hnsw->getCurrentElementCount());

    // Persist the database
    rag_hnsw->saveIndex(DB_CONFIG["index_path"]);
    delete rag_hnsw;

    // Persist the chunks metadata to disk
    std::ofstream chunks_json_file(DB_CONFIG["chunks_path"]);
    if (!chunks_json_file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
    } else {
        chunks_json_file << rag_json_root.dump(2); // Pretty print with 4 spaces indentation
        chunks_json_file.close();
    }

    // Update metadata
    meta_stats.model = eparams.model_name;
    // meta_stats.document_base_path = documents_path;
    meta_stats.dimension = eparams.n_dim;
    meta_stats.document_count = rag_docs.size();
    meta_stats.chunk_count = rag_chunks.size();
    meta_stats.chunk_size = eparams.chunk_size;
    meta_stats.index_ef_construction = ef_construction;
    meta_stats.index_M = M;
    meta_stats.index_ef_search = ef_search;
    meta_stats.embedding_time = embed_duration.count();
    meta_stats.verbose = eparams.verbose;

    generate_metadata_json(meta_stats, alldocs_stat, documents_path);

#if NOTNOW
    test_run(eparams);
#endif

    printf("BuildRag successfully built index database.\n");
    return 0;
}

#if NOT_NOW // Commented out for now

// Code for computing cosine similarity
#ifdef __AVX512F__

#include <iostream>
#include <vector>
#include <cmath>
#include <intrin.h>
#include <stdexcept>

class OptimizedCosineSimilarity {
public:
    static double calculate(const std::vector<double>& A,
                            const std::vector<double>& B) {
        if (A.size() != B.size() || A.empty()) {
            throw std::invalid_argument("Vectors must be non-empty and of equal length");
        }

        __m256d dot_product = _mm256_setzero_pd(); // AVX accumulator for dot product
        __m256d norm_A = _mm256_setzero_pd();      // AVX accumulator for vector A norm
        __m256d norm_B = _mm256_setzero_pd();      // AVX accumulator for vector B norm

        size_t i = 0;

        // Process 4 elements at a time using AVX instructions
        for (; i + 3 < A.size(); i += 4) {
            __m256d vecA = _mm256_loadu_pd(&A[i]);
            __m256d vecB = _mm256_loadu_pd(&B[i]);

            dot_product = _mm256_add_pd(dot_product, _mm256_mul_pd(vecA, vecB));
            norm_A = _mm256_add_pd(norm_A, _mm256_mul_pd(vecA, vecA));
            norm_B = _mm256_add_pd(norm_B, _mm256_mul_pd(vecB, vecB));
        }

        // Handle remaining elements that don't fit in SIMD processing
        double scalar_dot = 0.0, scalar_norm_A = 0.0, scalar_norm_B = 0.0;
        for (; i < A.size(); ++i) {
            scalar_dot += A[i] * B[i];
            scalar_norm_A += A[i] * A[i];
            scalar_norm_B += B[i] * B[i];
        }

        // Horizontal sum of SIMD registers
        double dot_sum = horizontal_sum(dot_product) + scalar_dot;
        double norm_A_sum = horizontal_sum(norm_A) + scalar_norm_A;
        double norm_B_sum = horizontal_sum(norm_B) + scalar_norm_B;

        if (norm_A_sum == 0.0 || norm_B_sum == 0.0) {
            throw std::invalid_argument("Vectors must be non-zero");
        }

        return dot_sum / (std::sqrt(norm_A_sum) * std::sqrt(norm_B_sum));
    }

private:
    // Helper function to horizontally sum AVX registers
    static double horizontal_sum(__m256d vec) {
        __m128d low = _mm256_castpd256_pd128(vec);
        __m128d high = _mm256_extractf128_pd(vec, 1);
        low = _mm_add_pd(low, high);
        __m128d high64 = _mm_unpackhi_pd(low, low);
        return _mm_cvtsd_f64(_mm_add_sd(low, high64));
    }
};

#if 0
int main() {
    std::vector<double> doc1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    std::vector<double> doc2 = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};

    try {
        double similarity = OptimizedCosineSimilarity::calculate(doc1, doc2);
        std::cout << "Optimized cosine similarity: " << similarity << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
#endif

#include <iostream>         // For standard I/O operations
#include <vector>           // For working with vectors
#include <cmath>            // For mathematical functions like sqrt
#include <stdexcept>       // For throwing exceptions

class CosineSimilarity {
public:
    // Calculate cosine similarity between two vectors
    static double calculate(const std::vector<double>& A,
                            const std::vector<double>& B) {
        // Ensure both vectors are of equal length and non-empty
        if (A.size() != B.size() || A.empty()) {
            throw std::invalid_argument("Vectors must be non-empty and of equal length");
        }

        double dot_product = 0.0;  // Stores the result of the dot product
        double norm_A = 0.0;      // Sum of squares for vector A
        double norm_B = 0.0;      // Sum of squares for vector B

        // Compute dot product and norms in a single loop for efficiency
        for (size_t i = 0; i < A.size(); ++i) {
            dot_product += A[i] * B[i];    // Element-wise multiplication
            norm_A += A[i] * A[i];        // Squaring elements of A
            norm_B += B[i] * B[i];        // Squaring elements of B
        }

        // Ensure neither vector has a zero magnitude to avoid division by zero
        if (norm_A == 0.0 || norm_B == 0.0) {
            throw std::invalid_argument("Vectors must be non-zero");
        }

        // Calculate cosine similarity
        return dot_product / (std::sqrt(norm_A) * std::sqrt(norm_B));
    }
};

#if 0
int main() {
    // Example vectors representing two documents or data points
    std::vector<double> doc1 = {1.0, 2.0, 3.0};  // First document vector
    std::vector<double> doc2 = {3.0, 4.0, 6.0};  // Second document vector

    try {
        // Calculate and print the cosine similarity
        double similarity = CosineSimilarity::calculate(doc1, doc2);
        std::cout << "Cosine similarity: " << similarity << std::endl;
    } catch (const std::exception& e) {
        // Handle errors gracefully
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
#endif

#else // __AVX512F__

float cosine_similarity(float *A, float *B)
{
    float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (auto i = 0; i < SIZE; ++i)
    {
        dot += A[i] * B[i];
        denom_a += A[i] * A[i];
        denom_b += B[i] * B[i];
    }
    return dot / (sqrt(denom_a) * sqrt(denom_b));
}

#endif // __AVX512F__

#endif // # if NOT_NOW