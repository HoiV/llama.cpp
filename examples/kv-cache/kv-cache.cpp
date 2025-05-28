#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...
#pragma warning (disable:4715) //  not all control paths return a value

#include "kv-cache.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace console {
    enum display_t {
        reset = 0,
        prompt,
        stats,
        error
    };

    void init(bool use_simple_io, bool use_advanced_display);
    void cleanup();
    void set_display(display_t display);
}

#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_BOLD          "\x1b[1m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"

int current_custom_prompt_index = 0;
std::vector<std::string> custom_prompts;
std::vector<std::string>::iterator custom_prompts_it;
std::string custom_prompts_output;
bool switch_prompt = false; // set true every time switch to a new prompt

namespace console {

    //
    // Console state
    //

    static bool      color_display    = false;
    static display_t current_display  = reset;
    static FILE*     out              = stdout;
    static void*     hConsole;

    //
    // Init and cleanup
    //

    void init(bool use_color) {
        color_display = use_color;

        // Windows-specific console initialization
        DWORD dwMode = 0;
        hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
            hConsole = GetStdHandle(STD_ERROR_HANDLE);
            if (hConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(hConsole, &dwMode))) {
                hConsole = nullptr;
            }
        }
        if (hConsole) {
            // Check conditions combined to reduce nesting
            if (color_display && !(dwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) &&
                !SetConsoleMode(hConsole, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
                color_display = false;
            }
            // Set console output codepage to UTF8
            SetConsoleOutputCP(CP_UTF8);
        }
    }

    void cleanup() {
        // Reset console display
        set_display(reset);
    }

    //
    // Display and IO
    //

    // Keep track of current display and only emit ANSI code if it changes
    void set_display(display_t display) {
        if (color_display && current_display != display) {
            fflush(stdout);
            switch(display) {
                case reset:
                    fprintf(out, ANSI_COLOR_RESET);
                    break;
                case stats:
                    fprintf(out, ANSI_COLOR_YELLOW);
                    break;
                case prompt:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_GREEN);
                    break;
                case error:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_RED);
            }
            current_display = display;
            fflush(out);
        }
    }
} // namespace console

bool processCustomPromptsFromFile(xbapp_params& xbparams) {
    std::ifstream cpfile(xbparams.custom_p_file);
    if (!cpfile.is_open()) {
        printf("[%s]: failed to open [%s]\n", __func__, xbparams.custom_p_file.c_str());
        return false;
    }

    std::string line;
    bool templatePromptMode = false;
    bool userPromptMode = false;
    custom_prompts.clear();
    std::string custom_template_prompt = "";

    // process CUSTOM_TEMPLATE_PROMPT
    while (std::getline(cpfile, line)) {
        if (line.find("CUSTOM_TEMPLATE_PROMPT") != std::string::npos) {
            templatePromptMode = true;
            continue;
        } else if (line.find("CUSTOM_PROMPT") != std::string::npos) {
            userPromptMode = true;
            continue;
        } else if (line.find("END_SECTION") != std::string::npos) {
            templatePromptMode = false;
            userPromptMode = false;
            continue;
        }

        if (templatePromptMode) {
            custom_template_prompt += line + '\n';
        } else if (userPromptMode) {
            custom_prompts.push_back(line);
        }
    }

    xbparams.custom_template_prompt = custom_template_prompt;

    cpfile.close();

    return true;
}

#ifdef _WIN32

#include <intrin.h>
namespace common {
    #include <xbox-cpu.h>
    #include <xbox-cpu.hpp>
}

#else // _WIN32

#define xb_set_process_affinity(n, m)
#define xb_set_optimal_process_affinity(n)

#endif // _WIN32

bool parse_cpu_mask(const std::string & mask, bool (&boolmask)[32]) {
    // Discard potential 0x prefix
    size_t start_i = 0;
    if (mask.length() >= 2 && mask.substr(0, 2) == "0x") {
        start_i = 2;
    }

    size_t num_digits = mask.length() - start_i;
    if (num_digits > 128) num_digits = 128;

    size_t end_i = num_digits + start_i;

    for (size_t i = start_i, n = (num_digits*4 - 1); i < end_i; i++, n-=4) {
        char c = mask.at(i);
        int8_t id = c;

        if ((c >= '0' && c <= '9')) {
            id -= '0';
        } else if (c >= 'a' && c <= 'f') {
            id -= 'a' - 10;
        } else if (c >= 'A' && c <= 'F') {
            id -= 'A' - 10;
        } else {
            fprintf(stderr, "Invalid hex character '%c' at position %d\n", c, int32_t(i));
            return false;
        }

        boolmask[n    ] = boolmask[n    ] || ((id & 8) != 0);
        boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
    }

    return true;
}

void print_system_info(xbapp_params& xb_params) {
    std::ostringstream os;

    os << "system_info: n_threads = " << xb_params.n_threads;
    if (xb_params.n_threads != -1) {
    os << " (n_batch = " << xb_params.n_batch << ")";
    }
#if defined(_WIN32) && (_WIN32_WINNT >= 0x0601) && !defined(__MINGW64__) // windows 7 and later
    // TODO: windows + arm64 + mingw64
    DWORD logicalProcessorCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    os << " / " << logicalProcessorCount << " | " << llama_print_system_info();
#else
    os << " / " << std::thread::hardware_concurrency() << " | " << llama_print_system_info();
#endif

    printf("\n%s: %s\n\n", __func__, os.str().c_str());
}

static void print_usage(int, char ** argv) {
    printf("\n%s: example usage:\n", __func__);
    printf("\n    %s -m model.gguf \n"
           "                [-n n_seqlen] [-t n_threads] [-cpf cpf_prompt] \n"
           "                [-pfc] [-ngl n_gpu_layers] [-vl 1|2|...|4] [-vv]\n"
#ifdef GGML_USE_OPENMP
           "                [-omp]\n"
#endif           
           "                [prompt...]\n", argv[0]);
}

void xbapp_log_callback(ggml_log_level level, const char * text, void * user_data) {
    GGML_UNUSED(text);

    ggml_log_level xbapp_log_level = (ggml_log_level)0 /* GGML_LOG_LEVEL_NONE */;
    if (user_data != nullptr) {
        xbapp_log_level = *(ggml_log_level *)user_data;
    }

    if (level == xbapp_log_level) {
        fputs(text, stdout);
    }
}

int64_t t0;
int main(int argc, char** argv) {
    // get default values
    xbapp_params xbparams;

    ggml_time_init();
    t0 = ggml_time_us();

    // parse command line args
    {
        int i = 1;
        for (; i < argc; i++) {
            if (strcmp(argv[i], "-paffin") == 0) {
                xbparams.process_affinity = true;
            } else if (strcmp(argv[i], "-cpf") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.custom_p_file = argv[++i];
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    xbparams.model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-repack") == 0) {
                if (i + 1 < argc) {
                    try {
                        int32_t repacking_mode = std::stoi(argv[++i]);
                        if (repacking_mode >= (int32_t) TENSOR_REPACKING_MODE_MAX) {
                            printf("%s: repacking mode %d is invalid (should be less than %d)\n",
                                __func__, repacking_mode, (uint32_t) TENSOR_REPACKING_MODE_MAX);
                            return 1;
                        } else {
                            xbparams.repacking_mode = (ggml_tensor_repacking_mode_t) repacking_mode;
                        }
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.n_seqlen = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.n_ngl = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-omp") == 0) {
                xbparams.openmp = true;
            } else if (strcmp(argv[i], "-pfc") == 0) {
                xbparams.pfc_mode = true;
            } else if (strcmp(argv[i], "-t") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.n_threads = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-C") == 0) {
                if (i + 1 < argc) {
                    if (!parse_cpu_mask(argv[++i], xbparams.cpumask)) {
                        fprintf(stderr, "error: failed to parse CPU mask: '%s'\n", argv[i]);
                        print_usage(argc, argv);
                        return 1;
                    } else {
                        xbparams.cpumask_present = true;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-vl") == 0) {
                if (i + 1 < argc) {
                    try {
                        xbparams.verbose_level = std::stoi(argv[++i]);
                    } catch (...) {
                        print_usage(argc, argv);
                        return 1;
                    }
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else if (strcmp(argv[i], "-vv") == 0) {
                xbparams.verbose_extra = true;
            } else {
                // single prompt starts here
                break;
            }
        }

        if (xbparams.model_path.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        if (i < argc) {
            // walk the single prompt until end
            xbparams.prompt = argv[i++];
            for (; i < argc; i++) {
                xbparams.prompt += " ";
                xbparams.prompt += argv[i];
            }
        }
    }

    printf("%s: Number of hw threads asked: %d\n", __func__, xbparams.n_threads);
    if (xbparams.n_threads <= 0) {
        int32_t n_threads = std::thread::hardware_concurrency();
        if (n_threads > 0) {
            n_threads = (n_threads <= 4) ? n_threads : (n_threads / 2);
        } else {
            n_threads = 4;
        }
        xbparams.n_threads = n_threads;
    }

    if (xbparams.repacking_mode != TENSOR_REPACKING_MODE_NONE) {
        ggml_set_tensor_repacking_mode(xbparams.repacking_mode);
    }

#ifdef GGML_USE_OPENMP
    xbparams.n_threads = MIN(xbparams.n_threads, omp_get_max_threads());
    if (xbparams.openmp) {
        printf("%s: OpenMP selected\n", __func__);
        // default mode if GGML_USE_OPENMP is defined
        // ggml_select_omp();
    }
#endif

    printf("%s: running with: %d threads\n", __func__, xbparams.n_threads);

    int64_t cpu_affinity_mask = 0;
    int32_t cpu_core_count_from_cpumask = 0;
    if (xbparams.cpumask_present) {
        for (int i = 0; i < 32; i++) {
            if (xbparams.cpumask[i]) {
                cpu_core_count_from_cpumask++;
                cpu_affinity_mask |= 1ull << i;
            }
        }
        printf("%s: CPU mask requested=[%0llX] - core count=[%d]\n", 
            __func__, cpu_affinity_mask, cpu_core_count_from_cpumask);
    }

    if (xbparams.cpumask_present && (cpu_core_count_from_cpumask >= xbparams.n_threads)) {
        cpu_affinity_mask = common::xb_set_process_affinity(xbparams.n_threads, cpu_affinity_mask);
    } else if (xbparams.process_affinity) {
        cpu_affinity_mask = common::xb_set_optimal_process_affinity(xbparams.n_threads);
    }
    printf("[%s]: Setting process affinity mask 0x%016llX\n", __func__, cpu_affinity_mask);

    console::init(true);
    printf("[%s]: processing cpf input file [%s]\n", __func__, xbparams.custom_p_file.c_str());
    processCustomPromptsFromFile(xbparams);
    custom_prompts_it = custom_prompts.begin();

    if (xbparams.verbose_extra ) {
        // default logging mode
    } else {
        // xbapp logging mode
        
        // map xbapp verbose level to current version of GGML definitions
        switch (xbparams.verbose_level) {
            case 0: break; // already 0 by default
            case 1: xbparams.log_level = GGML_LOG_LEVEL_INFO;  break; // info
            case 2: xbparams.log_level = GGML_LOG_LEVEL_WARN;  break; // warn
            case 3: xbparams.log_level = GGML_LOG_LEVEL_ERROR; break; // error
            case 4: xbparams.log_level = GGML_LOG_LEVEL_DEBUG; break; // debug
            default: break; // no match then default to no logging (0)
        }
        llama_log_set(xbapp_log_callback, &(xbparams.log_level));
    } 

    print_system_info(xbparams);

    // initialize the model
    if (slm_init(xbparams) != 0) {
        printf("%s: Error during slm_init()\n", __func__);
        return 1;
    }

    int prompt_index = 1;
    while (custom_prompts_it != custom_prompts.end())
    {
        // Create custom user prompt
        std::string& custom_prompt = ::trim(*custom_prompts_it);
        custom_prompt.erase(
            std::remove(custom_prompt.begin(), custom_prompt.end(), '\"'),
            custom_prompt.end());

        std::string full_prompt = ::trim(xbparams.custom_template_prompt);
        size_t pos = full_prompt.find("{message}");
        if (pos != std::string::npos) {
            full_prompt.replace(pos, std::string("{message}").length(), custom_prompt);
        }
        else {
            pos = 0;
        }

        if (xbparams.pfc_mode && !xbparams.pfx_shared.empty()) {
            xbparams.prompt = full_prompt.substr(pos);
        }
        else {
            // non pfc mode 
            xbparams.prompt = full_prompt;
        }

        console::set_display(console::prompt);
        printf("> Running with custom prompt => [%d/%zd]: [%s]\n",
            prompt_index++,
            custom_prompts.size(),
            custom_prompt.c_str());

        // reset for the output
        console::set_display(console::reset);

        // running an inference based on a prompt
        slm_inference(xbparams);

        xbparams.first_prompt = false;
        
        custom_prompts_it++;
    }

    xbparams.log_level = GGML_LOG_LEVEL_INFO;
    llama_log_set(xbapp_log_callback, &(xbparams.log_level));
    slm_terminate();
    
    console::set_display(console::stats);
    t0 = ggml_time_us() - t0;
    printf("\n\n total elapsed time %7.2fsec\n", (double)t0 / (1000. * 1000.));
#ifdef GGML_TENSOR_OP_PERF
    print_tensor_op_perf_data(t0);
#endif // GGML_TENSOR_OP_PERF

    console::cleanup();

    return 0;
}