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
// the screen - for speech or other logging purpose.
std::string tts_string;
std::string g_string_to_synthesize;
std::atomic<bool> g_synthesizingText(false);
shared_ptr<SpeechSynthesizer> g_synthesizer;

void StopTTS() {
    if (g_synthesizingText.load()) {
        g_synthesizer->StopSpeakingAsync().get();
    }
    g_synthesizingText.store(false);
}

void StartTTS() {
    if (!HasSpeechSynthesisVoice()) {
        return;
    }

    while (g_synthesizingText.load()) {
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
string SpeechSynthesisVoicePath;
string SpeechSynthesisVoiceName;

bool HasSpeechSynthesisVoice()
{
    if (SpeechSynthesisVoicePath.empty() || SpeechSynthesisVoiceName.empty())
    {
        cerr << "## ERROR: No speech synthesis voice specified.\n";
        return false;
    }
    return true;
}

// Creates an instance of an embedded speech config.
shared_ptr<EmbeddedSpeechConfig> CreateSpeechConfig()
{
    vector<string> paths;

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
    /*
    config->SetProperty(PropertyId::Speech_LogFilename, "SpeechSDK.log");
    */

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
    cout << "Embedded speech synthesis\n";
    cout << "  voice search path: " << (SpeechSynthesisVoicePath.empty() ? "(not set)" : SpeechSynthesisVoicePath) << endl;
    cout << "  voice name:        " << (SpeechSynthesisVoiceName.empty() ? "(not set)" : SpeechSynthesisVoiceName) << endl;

    return true;
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
#if 0 // Options to register to interesting events

    // Subscribes to events.
    synthesizer->SynthesisStarted += [](const SpeechSynthesisEventArgs& e)
    {
        UNUSED(e);
        // cout << "Synthesis started." << endl;
    };

    synthesizer->Synthesizing += [](const SpeechSynthesisEventArgs& e)
    {
        // cout << "Synthesizing, received an audio chunk of " << e.Result->GetAudioLength() << " bytes." << endl;
    };

    synthesizer->WordBoundary += [](const SpeechSynthesisWordBoundaryEventArgs& e)
    {
        // cout << "Word \"" << e.Text << "\" | "
        //     << "Text offset " << e.TextOffset << " | "
        //     // Unit of AudioOffset is tick (1 tick = 100 nanoseconds).
        //     << "Audio offset " << (e.AudioOffset + 5000) / 10000 << "ms"
        //     << endl;
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
    synthesizerThread.detach();
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
