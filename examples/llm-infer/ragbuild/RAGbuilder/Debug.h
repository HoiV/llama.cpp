/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    Debug.h

Abstract:

    This module defines the debug loging infrastructure for AIMX.
    Provides functions for logging debg information and errors.

Author:

    Rupo Zhang (rizhang) 03/21/2025

--*/

#pragma once

#include <string>
#include <windows.h>
#include <fstream>

namespace Debug {
    
// Initialize debug loging
bool
Initialize(
    _In_opt_ const std::wstring& logFilePath = L""
    );

// Log a debug mesage (narrow string)
void
Log(
    _In_ const std::string& message
    );

// Log a debug message (wide string)
void
Log(
    _In_ const std::wstring& message
    );

// Log an error message (narrow string)
void
LogError(
    _In_ const std::string& message
    );

// Log an error message (wide string)
void
LogError(
    _In_ const std::wstring& message
    );

// Get current log file path
std::wstring
GetLogFilePath();

// Shotdown logging
void
Shutdown();
    
} // namespace Debug
