#pragma warning (disable:4267) //  conversion from 'size_t' to 'int' ...
#pragma warning (disable:4715) //  not all control paths return a value

#include "kv-cache.h"

int current_custom_prompt_index = 0;
std::vector<std::string> custom_prompts;
std::vector<std::string>::iterator custom_prompts_it;
std::string custom_prompts_output;
bool switch_prompt = false; // set true every time switch to a new prompt

bool processCustomPromptsFromFile(gpt_params& params) {
    std::ifstream cpfile(params.custom_p_file);
    if (!cpfile.is_open()) {
        printf("[%s]: failed to open [%s]\n", __func__, params.custom_p_file.c_str());
        return false;
    }

    std::string line;
    bool templatePromptMode = false;
    bool userPromptMode = false;
    custom_prompts.clear();
    std::string custom_template_prompt = "";

    // process CUSTOM_TEMPLATE_PROMPT
    while (std::getline(cpfile, line)) {
        if (line == "CUSTOM_TEMPLATE_PROMPT") {
            templatePromptMode = true;
            continue;
        } else if (line == "CUSTOM_PROMPT") {
            userPromptMode = true;
            continue;
        } else if (line == "END_SECTION") {
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

    params.custom_template_prompt = custom_template_prompt;

    cpfile.close();

    return true;
}

int main(int argc, char** argv) {
    gpt_params params;

    if (argc == 1 || argv[1][0] == '-')
    {
        printf("usage: %s MODEL_PATH [#_threads] [CUSTOM_PROMPT_file] [pfc]\n", argv[0]);
        return 1;
    }

    if (argc >= 2) {
        params.model = argv[1];
    }

    if (argc >= 3) {
        unsigned int n_threads = std::stoi(argv[2]);
        if (n_threads <= 0) {
            n_threads = std::thread::hardware_concurrency();
            if (n_threads > 0) {
                n_threads = (n_threads <= 4) ? n_threads : (n_threads / 2);
            }
        }
        params.n_threads = n_threads;
    }

    if (argc >= 4) {
        params.custom_p_file = argv[3];
    }

    if (argc >= 5) {
        params.pfc_mode = true;
    }

    printf("[%s]: processing cpf input file [%s]\n", __func__, params.custom_p_file.c_str());
    processCustomPromptsFromFile(params);
    custom_prompts_it = custom_prompts.begin();

    // initialize the model
    if (slm_init(params) != 0) {
        printf("%s: Error during slm_init()\n", __func__);
        return 1;
    }

    int prompt_index = 1;
    while (custom_prompts_it != custom_prompts.end())
    {
        // Create custom user prompt
        std::string& custom_prompt = *custom_prompts_it;
        custom_prompt.erase(
            std::remove(custom_prompt.begin(), custom_prompt.end(), '\"'),
            custom_prompt.end());

        std::string full_prompt = params.custom_template_prompt;
        size_t pos = full_prompt.find("{message}");
        if (pos != std::string::npos) {
            full_prompt.replace(pos, std::string("{message}").length(), custom_prompt);
        }
        else {
            pos = 0;
        }

        if (params.pfc_mode) {
            params.prompt = full_prompt.substr(pos);
        }
        else {
            // non pfc mode 
            params.prompt = full_prompt;
        }

        printf("> Running with custom prompt => [%d/%zd]: [%s]\n",
            prompt_index++,
            custom_prompts.size(),
            custom_prompt.c_str());

        // running an inference based on a prompt
        slm_inference(params);

        params.first_prompt = false;
        
        custom_prompts_it++;
    }

    slm_terminate();

    return 0;
}