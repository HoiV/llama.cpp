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
std::vector<std::string> custom_settings;
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
}

bool processCustomPromptsFromFile(xbapp_params& xbparams) {
    std::ifstream cpfile(xbparams.custom_p_file);
    if (!cpfile.is_open()) {
        printf("[%s]: failed to open [%s]\n", __func__, xbparams.custom_p_file.c_str());
        return false;
    }

    std::string line;
    bool templatePromptMode = false;
    bool userPromptMode = false;
    bool settingPromptMode = false;
    custom_prompts.clear();
    custom_settings.clear();
    std::string custom_template_prompt = "";

    // process CUSTOM_TEMPLATE_PROMPT
    while (std::getline(cpfile, line)) {
        if (line.find("CUSTOM_TEMPLATE_PROMPT") != std::string::npos) {
            templatePromptMode = true;
            continue;
        } else if (line.find("CUSTOM_SETTING") != std::string::npos) {
            settingPromptMode = true;
            continue;
        } else if (line.find("CUSTOM_PROMPT") != std::string::npos) {
            userPromptMode = true;
            continue;
        } else if (line.find("END_SECTION") != std::string::npos) {
            templatePromptMode = false;
            userPromptMode = false;
            settingPromptMode = false;
            continue;
        }

        if (templatePromptMode) {
            custom_template_prompt += line + '\n';
        } else if (settingPromptMode) {
            custom_settings.push_back(line);
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

uint64_t l1d_cache_size = 48ull * 1024ull;
uint64_t l1i_cache_size = 32ull * 1024ull;
uint64_t l2_cache_size = 1024ull * 1024ull;
uint64_t l3_cache_size = 1024ull * 1024ull;

typedef struct {
    uint64_t mask;
    uint16_t group;
    uint16_t reserved[3];
} group_affinity_t;

void
xb_set_process_affinity (
    uint32_t n_threads,
    int64_t affinity_mask_requested
    )
{
#if defined(__x86_64__) || defined(_M_X64)

    //
    // Get the default rounding mode.
    //

    char * default_mode = "none";

    uint32_t mxcsr = _mm_getcsr();

    uint32_t round_mode = mxcsr & _MM_ROUND_MASK;

    switch (round_mode) {
    case _MM_ROUND_NEAREST:
        default_mode = "round nearest";
        break;

    case _MM_ROUND_DOWN:
        default_mode = "round_down";
        break;

    case _MM_ROUND_UP:
        default_mode = "round_up";
        break;

    case _MM_ROUND_TOWARD_ZERO:
        default_mode = "round_toward_zero";
        break;

    }

    // printf("mxcsr 0x%08lx, default rounding mode - %s\n", mxcsr, default_mode);

    //
    // Get number of logical processors per physical core and the maximum number of logical
    // processsors.
    //

    struct {
        uint32_t eax;
        uint32_t ebx;
        uint32_t ecx;
        uint32_t edx;
    } cpu_info;

    int64_t affinity_mask = affinity_mask_requested;

    if (affinity_mask_requested != 0) {
        goto set_affinity;
    }

    //
    // Get L1 instruction and data cache attributes.
    //

    __cpuid((int *)&cpu_info, 0x80000005);

//    printf("l1 d-cache line size %d\n", cpu_info.ecx & 0xff);
//    printf("l1 d-cache lines per tag %d\n", (cpu_info.ecx >> 8) & 0xff);
//    printf("l1 d-cache associativity %d\n", (cpu_info.ecx >> 16) & 0xff);

    l1d_cache_size = ((cpu_info.ecx >> 24) & 0xff) * 1024ull;
    // printf("l1 d-cache size in bytes %zd\n", l1d_cache_size);

//    printf("l1 i-cache line size %d\n", cpu_info.edx & 0xff);
//    printf("l1 i-cache lines per tag %d\n", (cpu_info.edx >> 8) & 0xff);
//    printf("l1 i-cache associativity %d\n", (cpu_info.edx >> 16) & 0xff);

    l1i_cache_size = ((cpu_info.edx >> 24) & 0xff) * 1024ull;
    // printf("l1 i-cache size in bytes %zd\n", l1i_cache_size);

    //
    // Get l2 and l3 cache sizes.
    //

    __cpuid((int *)&cpu_info, 0x80000006);

    l2_cache_size = ((cpu_info.ecx >> 16) & 0xffff) * 1024ull;
    // printf("l2 cache size in bytes %zd\n", l2_cache_size); 

//    l3_cache_size = ((cpu_info.edx >> 18) & 0x3fff); // * 1024ull;
//    printf("l3 cache size in bytes %zd\n", l3_cache_size); 

    //
    // Get logical processors per core.
    //

    // printf("n_threads specified %d\n", n_threads);
    __cpuid((int *)&cpu_info, 0x8000001e);
    const uint32_t logical_per_physical_core = ((cpu_info.ebx & 0x300) >> 8) + 1;
    // printf("number of logical processors per physical core %d\n", logical_per_physical_core);

    if (logical_per_physical_core == 1) {
        // printf("bypassing set process affinity - not SMT system\n");
        return;
    }

    __cpuid((int *)&cpu_info, 0x00000001);
    const uint32_t maximum_logical = (cpu_info.ebx & 0xff0000) >> 16;
    // printf("maximum number of logical processors %d\n", maximum_logical);

    //
    // Check the specified number of threads against the maximum logical processor count.
    //

    const uint32_t maximum_smt_threads = maximum_logical / 2;
    if ((n_threads & 1) || (n_threads > maximum_smt_threads)) {
        // printf("bypassing set process affinity - number threads odd or gt maximum logical / 2\n");
        return;
    }

    //
    // Get the current process group count.
    //

#if 0
    uint16_t group_array[4];
    uint16_t group_count = 4;

    if (GetProcessGroupAffinity(GetCurrentProcess(), &group_count, group_array)) {
        printf("GetProcessGroupAffinity succeeded with %d groups\n", group_count);
        if (group_count != 1) {
            printf("bypassing set affinity process because group count is greater than one\n");
            return;
        }

    } else {
        printf("GetProcessGroupAffinity failed\n");
        return;
    }
#endif // #if 0

    //
    // Set process affinity.
    //

    affinity_mask = ((1ull << (n_threads * 2)) - 1) & 0x55555555ull;

    //
    // It is known that the number of threads fits within the maximum smt set. If the
    // maximum smt set is less than or equal to 32, then the threads can be pushed
    // up to higher numbers threads which will remove them from contention issues
    // with clock and device interrupts.
    //

    if (maximum_smt_threads <= 32) {

        //
        // Compute the shift up such that the thread affinity straddles CCDs.
        //

        uint32_t half_shift = maximum_logical - (n_threads * 2);

        half_shift = ((half_shift / 2) + 1) & 0x1e;

        affinity_mask <<= half_shift;
    }

set_affinity:
    if (SetProcessAffinityMask(GetCurrentProcess(), affinity_mask)) {
        // printf("process group affinity set to 0x%08llx\n", affinity_mask);

    } else {
        // printf("failed to set process affinity mask\n");
    }


#else

    // printf("%s: set process affinity is only available for x86 architecture\n", __func__);

#endif // __x86_64__ || _M_X64_

    return;
}

#else

#define xb_set_process_affinity(n)

#endif // _WIN32

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
            if (strcmp(argv[i], "-affin") == 0) {
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

#ifdef GGML_USE_OPENMP
    xbparams.n_threads = MIN(xbparams.n_threads, omp_get_max_threads());
    if (xbparams.openmp) {
        printf("%s: OpenMP selected\n", __func__);
        // default mode if GGML_USE_OPENMP is defined
        // ggml_select_omp();
    }
#endif

    CPUInfo cinfo;
    cout << "CPU vendor = " << cinfo.vendor() << endl;
    cout << "CPU Brand String = " << cinfo.model() << endl;
    cout << "# of cores = " << cinfo.cores() << endl;
    cout << "# of logical cores = " << cinfo.logicalCpus() << endl;
    cout << "Is CPU Hyper threaded = " << cinfo.isHyperThreaded() << endl;

    if (cinfo.vendor().find("AMD") != std::string::npos) {
        if (cinfo.model().find("AMD Ryzen AI 9 HX 370") != std::string::npos) {
            printf("%s: Detected AMD Ryzen HX 370\n", __func__);
            xbparams.is_AMD_Ryzen_HX_370 = true;
        } else if (cinfo.model().find("AMD RYZEN AI MAX+ PRO 395") != std::string::npos) {
            printf("%s: Detected AMD Ryzen PRO 395\n", __func__);
            xbparams.is_AMD_Ryzen_PRO_395 = true;
        }
    }

    printf("%s: Actual using: %d threads\n", __func__, xbparams.n_threads);

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
    std::string full_prompt = ::trim(xbparams.custom_template_prompt);
    // locate the variant part of the prompt
    size_t setting_index = full_prompt.find("{CURRENT_SETTING}");
    if (setting_index == std::string::npos) {
        printf("%s: template prompt is not correctly formed for pfc mode - "
               "no \"{CURRENT_SETTING}\" identifier located\n", __func__);
        if (xbparams.pfc_mode) {
            return 1;
        }
    }

    while (custom_prompts_it != custom_prompts.end()) {
        // extract custom user prompt
        std::string& custom_prompt = *custom_prompts_it;
        console::set_display(console::prompt);
        printf("> Running with custom prompt => [%d/%zd]: [%s]\n",
            prompt_index++,
            custom_prompts.size(),
            custom_prompt.c_str());

        // reset for the output
        console::set_display(console::reset);

        // remove double quotes if any
        custom_prompt.erase(
            std::remove(custom_prompt.begin(), custom_prompt.end(), '\"'),
            custom_prompt.end());
        // trim any trailing spaces
        custom_prompt = ::trim(custom_prompt);

        // locate delimiter for custom setting within custom prompt
        size_t custom_setting_delim = custom_prompt.find(":");
        if (custom_setting_delim != std::string::npos) {
            // decode the custom setting (0-based)
            std::string custom_setting_str = custom_prompt.substr(0, custom_setting_delim);
            int custom_setting_number = std::stoi(custom_setting_str) - 1;
            // erase the setting number in the custom prompt (i.e. "1 : ...")
            custom_prompt.erase(0, (custom_setting_delim + 1));
            // printf("%s: custom prompt [%s]\n", __func__, custom_prompt.c_str());
            // printf("%s: custom setting [%s]\n", __func__, custom_settings[custom_setting_number].c_str());

            // preprend the setting in front of the custom prompt
            std::string tmp = custom_settings[custom_setting_number];
            // remove the numbering scheme for setting
            tmp.erase(0, 3); 
            // tmp += " } \n \"User\" : { \"";
            tmp += " \"";
            custom_prompt.insert(0, tmp);
            custom_prompt += " \" } \n <|end|> \n <|assistant|> \n You :";
        }

        // build the full prompt
        if (xbparams.pfc_mode && !xbparams.pfx_shared.empty()) {
            // for pfc mode the prompt is the part that keeps changing
            printf("%s: custom prompt [%s]\n", __func__, custom_prompt.c_str());
            xbparams.prompt = custom_prompt;
        } else {
            GGML_ASSERT(setting_index != std::string::npos);
            full_prompt.erase(setting_index);
            full_prompt.insert(setting_index, custom_prompt);

            // use the full prompt for non-pfc mode or pfc-mode with no cache present yet
            xbparams.prompt = full_prompt;
        }

        // running an inference based on a prompt
        slm_inference(xbparams);

        xbparams.first_prompt = false;
        
        custom_prompts_it++;
    }

    slm_terminate();
    
    console::set_display(console::stats);
    t0 = ggml_time_us() - t0;
    printf("\n\n total elapsed time %7.2fsec\n", (double)t0 / (1000. * 1000.));
#ifdef GGML_TENSOR_OP_PERF
    print_tensor_op_perf_data();
#endif // GGML_TENSOR_OP_PERF

    console::cleanup();

    return 0;
}