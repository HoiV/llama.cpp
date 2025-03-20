#pragma warning (disable:4100) // unref formal param
#pragma warning (disable:4127) // cond expression is constant
#pragma warning (disable:4242) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4244) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4245) // conversion from 'int' to 'hnswlib::vl_type'
#pragma warning (disable:4267) // conversion from 'int64_t' to 'int'
#pragma warning (disable:4505) // unref function with internal linkage removed

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <filesystem> // For file existence check

#include "llm-infer.h"
#include <json.hpp> // For JSON handling

using json = nlohmann::json;
using namespace std;
namespace fs = filesystem;

// Configuration paths
unordered_map<string, string> DB_CONFIG = {
    {"index_path", "./rag_index.bin"},
    {"chunks_path", "./chunks.json"},
    {"metadata_path", "./metadata.json"}
};

static bool ragdb_initialized = false;
std::vector<rag_entry> rag_entries;
hnswlib::HierarchicalNSW<float>* ragDB;

vector<rag_entry> retrieve_chunks(
    const model_params & eparams,
    const string& query, 
    int top_k = 3) {

    vector<rag_entry> rag_context;

    if (!ragdb_initialized) {
        printf("RAG Database not initialized - please call embed_initialize()\n");
        return rag_context;
    }

    try {
        // Generate query embedding
        vector<float> embeddings(eparams.n_dim, 0);
        if (!embed_encode_single(eparams, query, embeddings)) {
            printf("%s: error during encoding for query '%s'\n", __func__, query.c_str());
            return rag_context;
        }

        // Search for nearest chunks
        try {
            int k = min(top_k, static_cast<int>(rag_entries.size()));
            std::vector<std::pair<float, hnswlib::labeltype>> results = ragDB->searchKnnCloserFirst(embeddings.data(), k);

            for (auto item: results) {
                int idx = /* chunk index/label */ item.second;

                auto rag_item = rag_entries[idx];
                // printf("    [%4d]. Distance: %5.2f - [%zd]-<%.40s>\n", rag_item.id, item.first, idx, rag_item.textdata.c_str());
                if (idx != rag_item.id) {
                    printf("%s: Error during retrieval of rag item: id = %4d - expected value = %4d\n", 
                        __func__, idx, rag_item.id);
                }
                rag_context.push_back(rag_item);
            }

        } catch (const exception& e) {
            cerr << "Search failed: " << e.what() << endl;
            return rag_context;
        }

    } catch (const exception& e) {
        cerr << "Error in retrieve_chunks: " << e.what() << endl;
    }

    return rag_context;
}

bool RAG_initialize(model_params & eparams) {
    ragDB = nullptr;    

    string index_path = DB_CONFIG["index_path"];
    string chunks_path = DB_CONFIG["chunks_path"];
    string metadata_path = DB_CONFIG["metadata.json"];
    
    // Check for required files
    if (!fs::exists(index_path) || !fs::exists(chunks_path) /* || !fs::exists(metadata_path) */) {
        cout << "Error: Required database files not found in " << index_path << endl;
        return false;
    }

    // Load the chunks info
    ifstream chunks_json_file(chunks_path);
    json chunks_json;
    try {
        chunks_json = json::parse(chunks_json_file);
    } catch (const exception& e) {
        cerr << "Error reading chunk: " << e.what() << endl;
        return false;
    }

    for (const auto& item : chunks_json) {
        int id = item["id"];
        string source = item["source"];
        string text = item["text"];
        rag_entry entry = {id, source, text};
        rag_entries.push_back(entry);
    }

    // Get dimension from metadata
    string doc_count = "";
    ifstream metadata_file(metadata_path);
    if (metadata_file.good()) {
        try {
            json metadata = json::parse(metadata_file);
            if (metadata.contains("dimension")) {
                int meta_dimension = metadata["dimension"];
                if (meta_dimension != eparams.n_dim) {
                    cout << "Using dimension from metadata: " << meta_dimension << endl;
                    eparams.n_dim = meta_dimension;
                }
            }
            doc_count = metadata.contains("document_count") ? to_string(metadata["document_count"]) : "?";

        } catch (const exception& e) {
            cerr << "Error reading metadata: " << e.what() << endl;
            return false;
        }
        metadata_file.close();
    }

    cout <<"Loaded " + to_string(rag_entries.size()) + " chunks from " + doc_count + " documents" << endl;

    // load the database
    try {
        // Load index DB
        cout << "Loading index with dimension " << eparams.n_dim << endl;

        hnswlib::L2Space *rag_space = new hnswlib::L2Space(eparams.n_dim);
        ragDB = new hnswlib::HierarchicalNSW<float>(rag_space, index_path);
        cout << "RAG database loaded from: " << index_path << endl;
                   
    } catch (const exception& e) {
        cout << "Error loading database: " << string(e.what()) << endl;
        return false;
    }

    size_t max_elements = ragDB->getMaxElements();
    size_t cur_elementCount = ragDB->getCurrentElementCount();
        
    printf("[%s]: load vector DB '%s' - max_elms(%zd)/cur_elms(%zd)\n", __func__, index_path.c_str(),
        max_elements, cur_elementCount);

    // Set flag to indicate readiness for inference
    ragdb_initialized = true;

    return true;
}

static bool __llm_inference(
    model_params & sparams,
    const string & query) {

    bool ret = false;

    __try {
        ret = llm_inference(sparams);
        if (!ret) {
            printf("Evaluating query '%s' failed\n", query.c_str());
        }
    } 
    __except (GetExceptionCode() == EXCEPTION_ILLEGAL_INSTRUCTION ? 
        EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH) {
        printf("llm_inference(): Failed with invalid instruction exception \n"
               "                 (no AVX512 support with this processor)\n");
        return false;
    }

    return ret;
}

void run_inference(
    model_params & eparams, 
    model_params & sparams,
    string query) {

    vector<rag_entry> rag_context = retrieve_chunks(eparams, query);
    string full_prompt = 
        "<|system|>\nYou are an assistant for question-answering tasks. Use the following pieces of "
        "retrieved context to answer the question. If you don't know the answer, just say that you don't "
        "know. Use three sentences maximum and keep the answer concise. Do not generate more questions after "
        "the provided answer.\n"
        "Context: {context}\n"
        "<|end|>"
        "<|user|>"
        "Question: {user-input}\n"
        "<|end|>"
        "<|Assistant|>"
        "Answer:";

    string context_info = "";
    for (auto item : rag_context) {
        context_info += item.textdata + "\n";
    }

    const string context_marker = "{context}";
    size_t context_pos = full_prompt.find(context_marker);
    full_prompt.replace(context_pos, context_marker.length(), context_info);
    const string input_marker = "{user-input}";
    size_t input_pos = full_prompt.find(input_marker);
    full_prompt.replace(input_pos, input_marker.length(), query);

    sparams.prompt = full_prompt;
    // printf("Fullprompt = \n[%s]\n\n\n", sparams.prompt.c_str());

    printf("Prompt:\n[%s]\n\n>> Reply: ", query.c_str());
    if (!__llm_inference(sparams, query)) {
        printf("Failed token generation for query: %s\n", query.c_str());
        sparams.reply = "";
    }

    if (!sparams.streaming_reply) {
        printf("%s\n<<\n\n\n", sparams.reply.c_str());
    } else {
        printf("\n<<\n\n\n");
    }

    return;
}

int main() {

    // Initialize sentencepiece for embeddings 
    model_params eparams;
    eparams.model_name = "./models/all-minilm-l6-v2_f32.gguf";
    eparams.n_ctx = 512;
    
    // Initialize SLM for conversations
    model_params sparams;
    sparams.model_name = "./models/Phi-3-mini-4k-instruct-Q4_K_M.gguf";
    sparams.streaming_reply = 1;

    vector<string> queries = {
        "Who is Rupo Zhang",
        "Do you know anything about a person named Rupo Zhang?",
        "Please describe more about what you know about Shilpa Patil",
        "Who won the Nobel prize?"
    };

    // Grab config from metadata file if it exists to override current defaults
    ifstream metadata_json_file(DB_CONFIG["metadata_path"]);
    try {
            // override embedding model params list

            if (metadata_json_file.is_open()) {
                json config_json = json::parse(metadata_json_file);
                if (config_json.contains("model")) {
                    string config_model = config_json["model"];
                    if (eparams.model_name != config_model) {
                        // Override defaults if model name from metadata.json is different
                        cout << "- Switching to embedding model from metadata json: " << config_model << endl;
                        eparams.model_name = config_model;
                    }
                }
                if (config_json.contains("verbose")) {
                    int config_verbose = config_json["verbose"];
                    if (eparams.verbose != config_verbose) {
                        // Override defaults if setting from metadata.json is different
                        cout << "- Switching verbose setting as specified in metadata json: " << config_verbose << endl;
                        eparams.verbose = config_verbose;
                    }
                }
    
                // override SLM params list
    
                if (config_json.contains("model-slm")) {
                    string config_model_slm = config_json["model-slm"];
                    if (sparams.model_name != config_model_slm) {
                        // Override defaults if model name from metadata.json is different
                        cout << "- Switching to SLM model from metadata json: " << config_model_slm << endl;
                        sparams.model_name = config_model_slm;
                    }
                }
                if (config_json.contains("streaming-reply")) {
                    int config_streaming = config_json["streaming-reply"];
                    if (eparams.streaming_reply != config_streaming) {
                        // Override defaults if setting from metadata.json is different
                        cout << "- Switching streaming mode as specified in metadata json: " << config_streaming << endl;
                        sparams.streaming_reply = config_streaming;
                    }
                }
                if (config_json.contains("queries")) {
                    vector<string> config_queries;
                    vector<json> config_json_queries = config_json["queries"];
                    for (auto query: config_json_queries) {
                        if (query.contains("query")) {
                            config_queries.push_back(query["query"].get<string>().c_str());
                        }
                    }
                    if (config_queries.size() != 0) {
                        // Override default queries with these new queries from metadata.json
                        cout << "- Using [" << config_queries.size() << "] queries from metadata json" << endl;
                        queries = config_queries;
                    }
                }
                if (config_json.contains("force_cpu")) {
                    int config_force_cpu = config_json["force_cpu"];
                    if (sparams.force_cpu_mode != config_force_cpu) {
                        // Override defaults if setting from metadata.json is different
                        cout << "- Forcing CPU mode as specified in metadata json: " << config_force_cpu << endl;
                        sparams.force_cpu_mode = config_force_cpu;
                    }
                }
            }
            metadata_json_file.close();

    } catch (const exception& /* e */) {
        cerr << "Skipping out early reading config: " << DB_CONFIG["metadata_path"] << endl;
    }

    // Initialize sentencepiece for embeddings
    if (!embed_initialize(eparams)) {
        printf("Initialiazation of embedding model '%s' failed\n", eparams.model_name.c_str());
        return -1;
    }
    if (!RAG_initialize(eparams)) {
        printf("Failed to initialize RAG database\n");
        return -1;
    }

    // Initialize SLM
    if (!llm_initialize(sparams)) {
        printf("Initialization of SLM '%s' failed\n", sparams.model_name.c_str());
        return -1;
    }
    
    for (auto query: queries) {
        run_inference(eparams, sparams, query);
    }

    llm_terminate(sparams);
    embed_terminate();

    if (ragDB != nullptr) {
        ragdb_initialized = false;
        delete ragDB;
    }

    return 0;
}