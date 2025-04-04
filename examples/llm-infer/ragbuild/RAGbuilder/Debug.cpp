/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    Debug.cpp

Abstract:

    This module implements the debug logging infrastructure for AIMX.
    Handles file and debug output logging.

Author:

    Rupo Zhang (rizhang) 03/21/2025

--*/

#include "Debug.h"
#include "StringUtils.h"
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>

namespace Debug {
    
// Static variables with direct initialization
static std::wofstream s_logFile;
static std::wstring s_logFilePath;
static bool s_initialized = false;

/*++

Routine Description:

    Gets the current timestamp as a formatted string for logging.

Arguments:

    None.

Return Value:

    std::wstring containing the formatted timestamp.

--*/
std::wstring
GetTimestamp() 
{
    auto now = std::chrono::system_clock::now();
    auto time = static_cast<time_t>(std::chrono::system_clock::to_time_t(now));
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
        
    std::wstringstream ss;
    std::tm tm;
    localtime_s(&tm, &time);
    
    ss << std::put_time(&tm, L"%Y-%m-%d %H:%M:%S") 
        << L"." << std::setfill(L'0') << std::setw(3) << ms.count();
        
    return ss.str();
}

/*++

Routine Description:

    Initializes the debug logging system. Creates a log file if one is not specified.

Arguments:

    logFilePath - Optional path to the log file. If empty, a default path will be created.

Return Value:

    bool indicating success or failure of the initialization.

--*/
bool
Initialize(
    _In_opt_ const std::wstring& logFilePath
    ) 
{
    if (s_initialized) 
    {
        // Already initialized
        return true;
    }
    
    // If no path specified, create a default path
    if (logFilePath.empty()) 
    {
        wchar_t appPath[MAX_PATH];
        if (GetModuleFileNameW(NULL, appPath, MAX_PATH) > 0) 
        {
            std::wstring path(appPath);
            size_t lastSlash = path.find_last_of(L"\\");
            if (lastSlash != std::wstring::npos) 
            {
                s_logFilePath = path.substr(0, lastSlash + 1) + L"aimx_debug.log";
            } 
            else 
            {
                s_logFilePath = L"aimx_debug.log";
            }
        } 
        else 
        {
            s_logFilePath = L"aimx_debug.log";
        }
    } 
    else 
    {
        s_logFilePath = logFilePath;
    }
    
    // Open the log file
    s_logFile.open(s_logFilePath, std::ios::out | std::ios::app);
    if (!s_logFile.is_open()) 
    {
        return false;
    }
    
    s_initialized = true;
    
    // Write header to log file
    s_logFile << L"==================================================" << std::endl;
    s_logFile << L"AIMX Debug Log Started at " << GetTimestamp() << std::endl;
    s_logFile << L"==================================================" << std::endl;
    s_logFile.flush();
    
    // Log to debug output
    OutputDebugStringW(L"AIMX: Debug logging initialized\n");
    return true;
}

/*++

Routine Description:

    Writes a message to the log file with an optional error indicator.

Arguments:

    message - The message to write to the log file.
    isError - Flag indicating if this is an error message.

Return Value:

    None.

--*/
void
LogToFile(
    _In_ const std::wstring& message, 
    _In_ bool isError
    ) 
{
    if (!s_initialized || !s_logFile.is_open())
        return;

    s_logFile << GetTimestamp() << L" ";
    if (isError) 
    {
        s_logFile << L"ERROR: ";
    }
    s_logFile << message << std::endl;
    s_logFile.flush();
}

/*++

Routine Description:

    Logs a debug message using a narrow string.

Arguments:

    message - The debug message to log.

Return Value:

    None.

--*/
void
Log(
    _In_ const std::string& message
    ) 
{
    OutputDebugStringA(("AIMX: " + message + "\n").c_str());
    if (s_initialized)
    {
        LogToFile(Utf8ToWide(message), false);
    }
}

/*++

Routine Description:

    Logs a debug message using a wide string.

Arguments:

    message - The debug message to log.

Return Value:

    None.

--*/
void
Log(
    _In_ const std::wstring& message
    ) 
{
    OutputDebugStringW((L"AIMX: " + message + L"\n").c_str());
    if (s_initialized)
    {
        LogToFile(message, false);
    }
}

/*++

Routine Description:

    Logs an error message using a narrow string.

Arguments:

    message - The error message to log.

Return Value:

    None.

--*/
void
LogError(
    _In_ const std::string& message
    ) 
{
    OutputDebugStringA(("AIMX ERROR: " + message + "\n").c_str());
    if (s_initialized)
    {
        LogToFile(Utf8ToWide(message), true);
    }
}

/*++

Routine Description:

    Logs an error message using a wide string.

Arguments:

    message - The error message to log.

Return Value:

    None.

--*/
void
LogError(
    _In_ const std::wstring& message
    ) 
{
    OutputDebugStringW((L"AIMX ERROR: " + message + L"\n").c_str());
    if (s_initialized)
    {
        LogToFile(message, true);
    }
}

/*++

Routine Description:

    Gets the current log file path.

Arguments:

    None.

Return Value:

    std::wstring containing the path to the current log file.

--*/
std::wstring
GetLogFilePath() 
{
    return s_logFilePath;
}

/*++

Routine Description:

    Shuts down the debug logging system and closes the log file.

Arguments:

    None.

Return Value:

    None.

--*/
void
Shutdown() 
{
    if (s_initialized && s_logFile.is_open()) 
    {
        // Write footer to log file
        s_logFile << L"==================================================" << std::endl;
        s_logFile << L"AIMX Debug Log Ended at " << GetTimestamp() << std::endl;
        s_logFile << L"==================================================" << std::endl;
        
        // Close log file
        s_logFile.close();
        s_initialized = false;
    }
}

} // namespace Debug
