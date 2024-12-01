//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See https://aka.ms/csspeech/license for the full license information.
//

#include <iostream>
#include <thread>
#include <speechapi_cxx.h>

using namespace std;
using namespace Microsoft::CognitiveServices::Speech;
using namespace Microsoft::CognitiveServices::Speech::Audio;

#include <stdio.h>  /* for FILENAME_MAX */
#ifdef _MSC_VER
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#if !defined WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
#endif // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// extern void ListSpeechRecognitionModels();
// extern void SpeechRecognitionFromMicrophone();
// extern void ListSpeechSynthesisVoices();

void SpeechSynthesisToSpeaker(string);
bool HasSpeechSynthesisVoice();

// The following is a shadow copy of any string being transcibed to
// the screen - for speech or other AI needs (i.e. SLM)
std::string tts_string;

std::string g_string_to_synthesize;
std::atomic<bool> g_synthesizingText(false);
shared_ptr<SpeechRecognizer> g_recognizer = nullptr;
shared_ptr<SpeechSynthesizer> g_synthesizer = nullptr;
HANDLE g_synthThreadHandle = NULL;

void StopTTS() {
    // printf("%s: Stop TTS StopTTS()\n", __func__);

    // clear current buffer to allow new incoming text
    tts_string.clear();

    if (g_synthesizer == nullptr) {
        // the speech synthesizer was not initialized
        return;
    }

    if (!g_synthesizingText.load()) {
        return;
    }

    #if 0
    // TermninateThread() causes locking and long pauses
    if (g_synthThreadHandle != NULL) {
        printf("%s: terminating thread\n", __func__);
        TerminateThread(g_synthThreadHandle, 0);
    }
    #endif

    g_synthesizingText.store(false);
    g_synthesizer->StopSpeakingAsync().get();
}

void StartTTS() {
    if ((g_synthesizer == nullptr) || !HasSpeechSynthesisVoice()) {
        // The system has no speech synthesis capability
        return;
    }

    while (g_synthesizingText.load()) {
        // Stop any synthesizing speech
        // printf("%s: Stop TTS StartTTS()\n", __func__);
        StopTTS();
    }

    g_synthesizingText.store(true);
    g_string_to_synthesize = tts_string;

    // clear immediately to allow new incoming text
    tts_string.clear();

    if (!g_string_to_synthesize.empty() || 
        ((g_string_to_synthesize.size() == 1) && g_string_to_synthesize[0] != '\n')) {
        // printf("%s: ~>>>%s<<<~\n", __func__, g_string_to_synthesize.c_str());
        // Synthesize to speaker
        SpeechSynthesisToSpeaker(g_string_to_synthesize);
    }
}

// START OF CONFIGURABLE SETTINGS

// Embedded speech model license (text).
// This applies to embedded speech recognition and synthesis.
// It is presumed that all the customer's embedded speech models use the same license.
const string EmbeddedSpeechModelLicense = "YourEmbeddedSpeechModelLicense"; // or set EMBEDDED_SPEECH_MODEL_LICENSE

// Path to the local embedded speech recognition model(s) on the device file system.
// This may be a single model folder or a top-level folder for several models.
// Use an absolute path or a path relative to the application working folder.
// The path is recursively searched for model files.
// Files belonging to a specific model must be available as normal individual files in a model folder,
// not inside an archive, and they must be readable by the application process.
const string EmbeddedSpeechRecognitionModelPath = "YourEmbeddedSpeechRecognitionModelPath"; // or set EMBEDDED_SPEECH_RECOGNITION_MODEL_PATH

// Name of the embedded speech recognition model to be used for recognition.
// For example: "en-US" or "Microsoft Speech Recognizer en-US FP Model V8"
const string EmbeddedSpeechRecognitionModelName = "YourEmbeddedSpeechRecognitionModelName"; // or set EMBEDDED_SPEECH_RECOGNITION_MODEL_NAME

// Path to the local embedded speech synthesis voice(s) on the device file system.
// This may be a single voice folder or a top-level folder for several voices.
// Use an absolute path or a path relative to the application working folder.
// The path is recursively searched for voice files.
// Files belonging to a specific voice must be available as normal individual files in a voice folder,
// not inside an archive, and they must be readable by the application process.
const string EmbeddedSpeechSynthesisVoicePath = "YourEmbeddedSpeechSynthesisVoicePath"; // or set EMBEDDED_SPEECH_SYNTHESIS_VOICE_PATH

// Name of the embedded speech synthesis voice to be used for synthesis.
// For example: "en-US-JennyNeural" or "Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural)"
const string EmbeddedSpeechSynthesisVoiceName = "YourEmbeddedSpeechSynthesisVoiceName"; // or set EMBEDDED_SPEECH_SYNTHESIS_VOICE_NAME

// END OF CONFIGURABLE SETTINGS

// Embedded speech recognition default input audio format settings.
// In addition, little-endian signed integer samples are required.
uint32_t GetEmbeddedSpeechSamplesPerSecond() { return 16000; }  // or 8000
uint8_t GetEmbeddedSpeechBitsPerSample() { return 16; }         // DO NOT MODIFY; no other format supported
uint8_t GetEmbeddedSpeechChannels() { return 1; }               // DO NOT MODIFY; no other format supported

// Get a setting value from environment or defaults.
const string GetSetting(const char* environmentVariableName, const string& defaultValue)
{
#pragma warning(suppress : 4996) // getenv
    auto value = getenv(environmentVariableName);
    return value ? value : defaultValue;
}

// These are set in VerifySettings() after some basic verification.
string SpeechModelLicense;
string SpeechRecognitionModelPath;
string SpeechRecognitionModelName;
string SpeechSynthesisVoicePath;
string SpeechSynthesisVoiceName;

// Utility functions for main menu.
bool HasSpeechRecognitionModel()
{
    if (SpeechRecognitionModelPath.empty() || SpeechRecognitionModelName.empty())
    {
        // cerr << "## ERROR: No speech recognition model specified.\n";
        return false;
    }
    return true;
}

bool HasSpeechSynthesisVoice()
{
    if (SpeechSynthesisVoicePath.empty() || SpeechSynthesisVoiceName.empty())
    {
        // cerr << "## ERROR: No speech synthesis voice specified.\n";
        return false;
    }
    return true;
}

// Creates an instance of an embedded speech config.
shared_ptr<EmbeddedSpeechConfig> CreateSpeechConfig()
{
    vector<string> paths;

    // Add paths for offline data.
    if (!SpeechRecognitionModelPath.empty())
    {
        paths.push_back(SpeechRecognitionModelPath);
    }
    if (!SpeechSynthesisVoicePath.empty())
    {
        paths.push_back(SpeechSynthesisVoicePath);
    }

    if (paths.size() == 0)
    {
        cerr << "## ERROR: No model path(s) specified.\n";
        return nullptr;
    }

    // Note, if there is only one path then you can also use EmbeddedSpeechConfig::FromPath(string).
    // All paths must be valid directory paths on the file system, otherwise e.g. initialization of
    // embedded speech synthesis will fail.
    auto config = EmbeddedSpeechConfig::FromPaths(paths);

    // Enable Speech SDK logging. If you want to report an issue, include this log with the report.
    // If no path is specified, the log file will be created in the program default working folder.
    // If a path is specified, make sure that it is writable by the application process.
    //
    config->SetProperty(PropertyId::Speech_LogFilename, "SpeechSDK.log");
    //

    if (!SpeechRecognitionModelName.empty())
    {
        // Mandatory configuration for embedded speech (and intent) recognition.
        config->SetSpeechRecognitionModel(SpeechRecognitionModelName, SpeechModelLicense);
    }

    if (!SpeechSynthesisVoiceName.empty())
    {
        // Mandatory configuration for embedded speech synthesis.
        config->SetSpeechSynthesisVoice(SpeechSynthesisVoiceName, SpeechModelLicense);
        if (SpeechSynthesisVoiceName.find("Neural") != string::npos)
        {
            // Embedded neural voices only support 24kHz sample rate.
            config->SetSpeechSynthesisOutputFormat(SpeechSynthesisOutputFormat::Riff24Khz16BitMonoPcm);
        }
    }

    // Disable profanity masking.
    /*
    config->SetProfanity(ProfanityOption::Raw);
    */

    return config;
}


// Do some basic verification of embedded speech settings.
bool VerifySettings()
{
    vector<char> cwd;
    cwd.resize(FILENAME_MAX);

    if (GetCurrentDir(cwd.data(), FILENAME_MAX))
    {
        cout << "Current working directory: " << cwd.data() << endl;
    }
    else
    {
        cout << "## WARNING: Cannot get the current working directory, errno=" << errno << endl;
    }

    SpeechModelLicense = GetSetting("EMBEDDED_SPEECH_MODEL_LICENSE", EmbeddedSpeechModelLicense);
    if (SpeechModelLicense.empty() || SpeechModelLicense.compare("YourEmbeddedSpeechModelLicense") == 0)
    {
        cerr << "## ERROR: The embedded speech model license is not set.\n";
        return false;
    }

    SpeechRecognitionModelPath = GetSetting("EMBEDDED_SPEECH_RECOGNITION_MODEL_PATH", EmbeddedSpeechRecognitionModelPath);
    if (SpeechRecognitionModelPath.compare("YourEmbeddedSpeechRecognitionModelPath") == 0)
    {
        SpeechRecognitionModelPath.clear();
    }
    SpeechRecognitionModelName = GetSetting("EMBEDDED_SPEECH_RECOGNITION_MODEL_NAME", EmbeddedSpeechRecognitionModelName);
    if (SpeechRecognitionModelName.compare("YourEmbeddedSpeechRecognitionModelName") == 0)
    {
        SpeechRecognitionModelName.clear();
    }

    SpeechSynthesisVoicePath = GetSetting("EMBEDDED_SPEECH_SYNTHESIS_VOICE_PATH", EmbeddedSpeechSynthesisVoicePath);
    if (SpeechSynthesisVoicePath.compare("YourEmbeddedSpeechSynthesisVoicePath") == 0)
    {
        SpeechSynthesisVoicePath.clear();
    }
    SpeechSynthesisVoiceName = GetSetting("EMBEDDED_SPEECH_SYNTHESIS_VOICE_NAME", EmbeddedSpeechSynthesisVoiceName);
    if (SpeechSynthesisVoiceName.compare("YourEmbeddedSpeechSynthesisVoiceName") == 0)
    {
        SpeechSynthesisVoiceName.clear();
    }

    // Find an embedded speech recognition model based on the name.
    if (!SpeechRecognitionModelPath.empty() && !SpeechRecognitionModelName.empty())
    {
        auto config = EmbeddedSpeechConfig::FromPath(SpeechRecognitionModelPath);
        auto models = config->GetSpeechRecognitionModels();

        auto result =
            find_if(models.begin(), models.end(), [&](shared_ptr<SpeechRecognitionModel> model)
                {
                    return model->Name.compare(SpeechRecognitionModelName) == 0 || model->Locales[0].compare(SpeechRecognitionModelName) == 0;
                });

        if (result == models.end())
        {
            cout << "## WARNING: Cannot locate an embedded speech recognition model \"" << SpeechRecognitionModelName << "\"\n";
        }
    }

    // Find an embedded speech synthesis voice based on the name.
    if (!SpeechSynthesisVoicePath.empty() && !SpeechSynthesisVoiceName.empty())
    {
        auto config = EmbeddedSpeechConfig::FromPath(SpeechSynthesisVoicePath);
        auto synthesizer = SpeechSynthesizer::FromConfig(config, nullptr);

        bool found = false;
        const auto voicesList = synthesizer->GetVoicesAsync("").get();

        if (voicesList->Reason == ResultReason::VoicesListRetrieved)
        {
            const auto& voices = voicesList->Voices;
            auto result =
                find_if(voices.begin(), voices.end(), [&](shared_ptr<VoiceInfo> voice)
                    {
                        return voice->Name.compare(SpeechSynthesisVoiceName) == 0 || voice->ShortName.compare(SpeechSynthesisVoiceName) == 0;
                    });

            if (result != voices.end())
            {
                found = true;
            }
        }

        if (!found)
        {
            cout << "## WARNING: Cannot locate an embedded speech synthesis voice \"" << SpeechSynthesisVoiceName << "\"\n";
        }
    }

    cout << "Embedded speech recognition\n";
    cout << "  model search path: " << (SpeechRecognitionModelPath.empty() ? "(not set)" : SpeechRecognitionModelPath) << endl;
    cout << "  model name:        " << (SpeechRecognitionModelName.empty() ? "(not set)" : SpeechRecognitionModelName) << endl;
    cout << "Embedded speech synthesis\n";
    cout << "  voice search path: " << (SpeechSynthesisVoicePath.empty() ? "(not set)" : SpeechSynthesisVoicePath) << endl;
    cout << "  voice name:        " << (SpeechSynthesisVoiceName.empty() ? "(not set)" : SpeechSynthesisVoiceName) << endl;

    return true;
}

// Lists available embedded speech recognition models.
void ListSpeechRecognitionModels()
{
    // Creates an instance of an embedded speech config.
    auto speechConfig = CreateSpeechConfig();
    if (!speechConfig)
    {
        return;
    }

    // Gets a list of models.
    auto models = speechConfig->GetSpeechRecognitionModels();

    if (!models.empty())
    {
        cout << "Models found:" << endl;
        for (const auto& model : models)
        {
            cout << model->Name << endl;
            cout << " Locale(s): ";
            for (const auto& locale : model->Locales)
            {
                cout << locale << " ";
            }
            cout << endl;
            cout << " Path:      " << model->Path << endl;
        }

        // To find a model that supports a specific locale, for example:
        /*
        auto locale = "en-US";
        auto found =
            find_if(models.begin(), models.end(), [&](shared_ptr<SpeechRecognitionModel> model)
                {
                    return model->Locales[0].compare(locale) == 0;
                });
        if (found != models.end())
        {
            cout << "Found " << locale << " model: " << (*found)->Name << endl;
        }
        */
    }
    else
    {
        cerr << "No models found. Either the path is not valid or the format of model(s) is unknown." << endl;
    }
}

// Lists available embedded speech synthesis voices.
void ListSpeechSynthesisVoices()
{
    // Creates an instance of an embedded speech config.
    auto speechConfig = CreateSpeechConfig();
    if (!speechConfig)
    {
        return;
    }

    // Creates a speech synthesizer.
    auto synthesizer = SpeechSynthesizer::FromConfig(speechConfig, nullptr);

    // Gets a list of voices.
    auto result = synthesizer->GetVoicesAsync("").get();

    if (result->Reason == ResultReason::VoicesListRetrieved)
    {
        auto getGenderString = [](auto gender)
        {
            if (gender == SynthesisVoiceGender::Female) {
                return "Female";
            } else if (gender == SynthesisVoiceGender::Male) {
                return "Male";
            } else { // SynthesisVoiceGender::Unknown
                return "Unknown";
            }
        };

        cout << "Voices found:" << endl;
        for (const auto& voice : result->Voices)
        {
            cout << voice->Name << endl;
            cout << " Gender: " << getGenderString(voice->Gender) << endl;
            cout << " Locale: " << voice->Locale << endl;
            cout << " Path:   " << voice->VoicePath << endl;
        }

        // To find a voice that supports a specific locale, for example:
        /*
        const auto& voices = result->Voices;
        auto locale = "en-US";
        auto found =
            find_if(voices.begin(), voices.end(), [&](shared_ptr<VoiceInfo> voice)
                {
                    return voice->Locale.compare(locale) == 0;
                });
        if (found != voices.end())
        {
            cout << "Found " << locale << " voice: " << (*found)->Name << endl;
        }
        */
    }
    else if (result->Reason == ResultReason::Canceled)
    {
        cerr << "CANCELED: ErrorDetails=\"" << result->ErrorDetails << "\"" << endl;
    }
}

void SynthesizeSpeech(string textInput, shared_ptr<SpeechSynthesizer> synthesizer)
{
#if 0

    // Subscribes to events.
    synthesizer->SynthesisStarted += [](const SpeechSynthesisEventArgs& e)
    {
        UNUSED(e);
        // cout << "Synthesis started." << endl;
        if (!g_synthesizingText.load()) {
            //printf("%s: Stop TTS SynthesisStarted\n", __func__);
            g_synthesizer->StopSpeakingAsync().get();
        }
    };

    synthesizer->Synthesizing += [](const SpeechSynthesisEventArgs& e)
    {
        // cout << "Synthesizing, received an audio chunk of " << e.Result->GetAudioLength() << " bytes." << endl;
        if (!g_synthesizingText.load()) {
            //printf("%s: Stop TTS Synthesizing\n", __func__);
            g_synthesizer->StopSpeakingAsync().get();
        }
    };

    synthesizer->WordBoundary += [](const SpeechSynthesisWordBoundaryEventArgs& e)
    {
        // cout << "Word \"" << e.Text << "\" | "
        //     << "Text offset " << e.TextOffset << " | "
        //     // Unit of AudioOffset is tick (1 tick = 100 nanoseconds).
        //     << "Audio offset " << (e.AudioOffset + 5000) / 10000 << "ms"
        //     << endl;
        if (!g_synthesizingText.load()) {
            //printf("%s: Stop TTS WordBoundary\n", __func__);
            g_synthesizer->StopSpeakingAsync().get();
        }
    };

#endif

    {
        // Synthesizes text to speech.
        auto result = synthesizer->SpeakTextAsync(textInput).get();

        // To adjust the rate or volume, use SpeakSsmlAsync with prosody tags instead of SpeakTextAsync.
        /*
        const auto ssmlSynthBegin = "<speak version='1.0' xml:lang='en-US' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='http://www.w3.org/2001/mstts'>";
        const auto ssmlSynthEnd = "</speak>";
        stringstream ssml;

        // Range of rate is -100% to 100%, 100% means 2x faster.
        ssml << ssmlSynthBegin << "<prosody rate='50%'>" << text << "</prosody>" << ssmlSynthEnd;
        std::cout << "Synthesizing with rate +50%" << endl;
        auto result1 = synthesizer->SpeakSsmlAsync(ssml.str()).get();
        ssml.str("");

        // Range of volume is -100% to 100%, 100% means 2x louder.
        ssml << ssmlSynthBegin << "<prosody volume='50%'>" << text << "</prosody>" << ssmlSynthEnd;
        std::cout << "Synthesizing with volume +50%" << endl;
        auto result2 = synthesizer->SpeakSsmlAsync(ssml.str()).get();
        */

        if (result->Reason == ResultReason::SynthesizingAudioCompleted)
        {
            // cout << "Synthesis completed for text \"" << text << "\"" << endl;

            // See where the result came from, cloud (online) or embedded (offline)
            // speech synthesis.
            // This can change during a session where HybridSpeechConfig is used.
            /*
            cout << "Synthesis backend: " << result->Properties.GetProperty(PropertyId::SpeechServiceResponse_SynthesisBackend) << endl;
            */

            // To save the output audio to a WAV file:
            /*
            auto audioDataStream = AudioDataStream::FromResult(result);
            audioDataStream->SaveToWavFile("SynthesizedSpeech.wav");
            */

            // To read the output audio for e.g. processing it in memory:
            /*
            audioDataStream->SetPosition(0); // reset the stream position

            vector<uint8_t> buffer(16000);
            uint32_t readBytes = 0;
            uint32_t totalBytes = 0;

            while ((readBytes = audioDataStream->ReadData(buffer.data(), static_cast<uint32_t>(buffer.size()))) > 0)
            {
                cout << "Read " << readBytes << " bytes" << endl;
                totalBytes += readBytes;
            }
            cout << "Total " << totalBytes << " bytes" << endl;
            */
        }
        else if (result->Reason == ResultReason::Canceled)
        {
            auto cancellation = SpeechSynthesisCancellationDetails::FromResult(result);
            cout << "CANCELED: Reason=" << int(cancellation->Reason) << endl;

            if (cancellation->Reason == CancellationReason::Error)
            {
                cerr << "CANCELED: ErrorCode=" << int(cancellation->ErrorCode) << endl;
                cerr << "CANCELED: ErrorDetails=\"" << cancellation->ErrorDetails << "\"" << endl;
            }
        }
    }
}

// Synthesizes speech using embedded speech config and the system default speaker device.
void SpeechSynthesisToSpeaker(string textInput)
{
    thread synthesizerThread(SynthesizeSpeech, textInput, g_synthesizer);
    g_synthThreadHandle = synthesizerThread.native_handle();

    // Let the thread go and resume
    synthesizerThread.detach();
}

void RecognizeSpeech(shared_ptr<SpeechRecognizer> recognizer, bool useKeyword, bool waitForUser)
{
    promise<void> recognitionEnd;

    // Subscribes to events.
    recognizer->Recognizing += [](const SpeechRecognitionEventArgs& e)
    {
        // Intermediate result (hypothesis).
        if (e.Result->Reason == ResultReason::RecognizingSpeech)
        {
            cout << "Recognizing:" << e.Result->Text << endl;
        }
        else if (e.Result->Reason == ResultReason::RecognizingKeyword)
        {
            // ignored
        }
    };

    recognizer->Recognized += [](const SpeechRecognitionEventArgs& e)
    {
        if (e.Result->Reason == ResultReason::RecognizedKeyword)
        {
            // Keyword detected, speech recognition will start.
            cout << "KEYWORD: Text=" << e.Result->Text << endl;
        }
        else if (e.Result->Reason == ResultReason::RecognizedSpeech)
        {
            // Final result. May differ from the last intermediate result.
            cout << "RECOGNIZED: Text=" << e.Result->Text << endl;

            // See where the result came from, cloud (online) or embedded (offline)
            // speech recognition.
            // This can change during a session where HybridSpeechConfig is used.
            /*
            cout << "Recognition backend: " << e.Result->Properties.GetProperty(PropertyId::SpeechServiceResponse_RecognitionBackend) << endl;
            */

            // Recognition results in JSON format.
            //
            // Offset and duration values are in ticks, where a single tick
            // represents 100 nanoseconds or one ten-millionth of a second.
            //
            // To get word level detail, set the output format to detailed.
            // See EmbeddedSpeechRecognitionFromWavFile() in this source file
            // for a configuration example.
            //
            // If an embedded speech recognition model does not support word
            // timing, the word offset and duration values are always 0, and the
            // phrase offset and duration only indicate a time window inside of
            // which the phrase appeared, not the accurate start and end of speech.
            /*
            string jsonResult = e.Result->Properties.GetProperty(PropertyId::SpeechServiceResponse_JsonResult);
            cout << "JSON result: " << jsonResult << endl;
            */
            // For parsing and better presentation, use e.g. nlohmann/json.
            /*
            auto json = nlohmann::json::parse(jsonResult);
            cout << json.dump(4) << endl;

            if (json.contains("NBest")) // detailed results
            {
                auto best = json["NBest"].at(0);
                if (best.contains("Words")) // word level detail
                {
                    for (const auto& word : best["Words"])
                    {
                        cout << "Word: " << word["Word"] << " | "
                            << "Offset: " << word["Offset"] / 10000 << "ms | "
                            << "Duration: " << word["Duration"] / 10000 << "ms" << endl;
                    }
                }
            }
            */
        }
        else if (e.Result->Reason == ResultReason::NoMatch)
        {
            // NoMatch occurs when no speech phrase was recognized.
            auto reason = NoMatchDetails::FromResult(e.Result)->Reason;
            cout << "NO MATCH: Reason=";
            switch (reason)
            {
            case NoMatchReason::NotRecognized:
                // Input audio was not silent but contained no recognizable speech.
                cout << "NotRecognized" << endl;
                break;
            case NoMatchReason::InitialSilenceTimeout:
                // Input audio was silent and the (initial) silence timeout expired.
                // In continuous recognition this can happen multiple times during
                // a session, not just at the very beginning.
                cout << "InitialSilenceTimeout" << endl;
                break;
            default:
                // Other reasons are not supported in embedded speech at the moment.
                cout << int(reason) << endl;
                break;
            }
        }
    };

    recognizer->Canceled += [](const SpeechRecognitionCanceledEventArgs& e)
    {
        switch (e.Reason)
        {
        case CancellationReason::EndOfStream:
            // Input stream was closed or the end of an input file was reached.
            cout << "CANCELED: EndOfStream" << endl;
            break;

        case CancellationReason::Error:
            // NOTE: In case of an error, do not use the same recognizer for recognition anymore.
            cerr << "CANCELED: ErrorCode=" << int(e.ErrorCode) << endl;
            cerr << "CANCELED: ErrorDetails=\"" << e.ErrorDetails << "\"" << endl;
            break;

        default:
            cout << "CANCELED: Reason=" << int(e.Reason) << endl;
            break;
        }
    };

    recognizer->SessionStarted += [](const SessionEventArgs& e)
    {
        UNUSED(e);
        cout << "Session started." << endl;
    };

    recognizer->SessionStopped += [&recognitionEnd](const SessionEventArgs& e)
    {
        UNUSED(e);
        cout << "Session stopped." << endl;
        recognitionEnd.set_value();
    };

    if (useKeyword) {
        // Creates an instance of a keyword recognition model.
        auto keywordModel = KeywordRecognitionModel::FromFile("./bZM_keyword.table");

        // Starts the following routine:
        // 1. Listen for a keyword in input audio. There is no timeout.
        //    Speech that does not start with the keyword is ignored.
        // 2. If the keyword is spotted, start normal speech recognition.
        // 3. After a recognition result (that always includes at least
        //    the keyword), go back to step 1.
        // Steps 1-3 repeat until StopKeywordRecognitionAsync() is called.
        recognizer->StartKeywordRecognitionAsync(keywordModel).get();

        // Wait for the user to press Enter
        cin.get();

        // Stops recognition.
        recognizer->StopKeywordRecognitionAsync().get();
    
    } else {
        // Start continuous recognition
        recognizer->StartContinuousRecognitionAsync().get();

        if (waitForUser) {
            cin.get();

        } else {
            recognitionEnd.get_future().get();
        }

        // Stops recognition.
        recognizer->StopContinuousRecognitionAsync().get();
    }
}

// Recognizes speech using embedded speech config and the system default microphone device.
void SpeechRecognitionFromMicrophone()
{
    auto useKeyword = false;
    auto waitForUser = true;

    auto speechConfig = CreateSpeechConfig();
    auto audioConfig = AudioConfig::FromDefaultMicrophoneInput();

    g_recognizer = SpeechRecognizer::FromConfig(speechConfig, audioConfig);
    RecognizeSpeech(g_recognizer, useKeyword, waitForUser);
}

// Recognizes speech using embedded speech config and the system default microphone device.
// Recognition is triggered with a keyword.
void SpeechRecognitionWithKeywordFromMicrophone()
{
    auto useKeyword = true;
    auto waitForUser = true;

    auto speechConfig = CreateSpeechConfig();
    auto audioConfig = AudioConfig::FromDefaultMicrophoneInput();

    g_recognizer = SpeechRecognizer::FromConfig(speechConfig, audioConfig);
    RecognizeSpeech(g_recognizer, useKeyword, waitForUser);
}

int InitializeSpeechModels()
{
    try
    {
        if (VerifySettings() != true)
        {
            return 1;
        }

        auto speechConfig = CreateSpeechConfig();
        auto audioConfig = AudioConfig::FromDefaultSpeakerOutput();

        g_synthesizer = SpeechSynthesizer::FromConfig(speechConfig, audioConfig);
    }
    catch (const exception& e)
    {
        cerr << "Exception caught: " << e.what() << endl;
        return 2;
    }

    return 0;
}
