/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Module Name:

    StringUtils.h

Abstract:

    This module defines string utility functions for the AIMX application.
    Provides functions for string conversion and manipulation.

Author:

    Rupo Zhang (rizhang) 03/21/2025

--*/

#pragma once

#include <string>
#include <windows.h>

/*++

Routine Description:

    Converts a wide (UTF-16) string to a UTF-8 encoded string.

Arguments:

    wide - The wide string to convert to UTF-8.

Return Value:

    std::string containing the UTF-8 encoded representation of the input string.

--*/
inline
std::string
WideToUtf8(
    _In_ const std::wstring& wide
    )
{
    if (wide.empty()) return std::string();
    
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wide.data(), (int)wide.size(), 
                                          NULL, 0, NULL, NULL);
    std::string utf8(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), (int)wide.size(), 
                        &utf8[0], size_needed, NULL, NULL);
    return utf8;
}

/*++

Routine Description:

    Converts a UTF-8 encoded string to a wide (UTF-16) string.

Arguments:

    utf8 - The UTF-8 string to convert to wide string.

Return Value:

    std::wstring containing the wide (UTF-16) representation of the input string.

--*/
inline
std::wstring
Utf8ToWide(
    _In_ const std::string& utf8
    )
{
    if (utf8.empty()) return std::wstring();
    
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), (int)utf8.size(), 
                                          NULL, 0);
    std::wstring wide(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), (int)utf8.size(), 
                        &wide[0], size_needed);
    return wide;
}

/*++

Routine Description:

    Escapes a string for safe inclusion in JSON, handling special characters
    like quotes, backslashes, and control characters according to JSON spec.

Arguments:

    input - The string to escape for JSON inclusion.

Return Value:

    std::string containing the properly escaped JSON string.

--*/
inline
std::string
EscapeJsonString(
    _In_ const std::string& input
    )
{
    std::string output;
    output.reserve(input.length() * 2);
    
    for (char ch : input) {
        switch (ch) {
            case '\"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 32) {
                    char buffer[8];
                    sprintf_s(buffer, "\\u%04x", ch);
                    output += buffer;
                } else {
                    output += ch;
                }
                break;
        }
    }
    
    return output;
}

/*++

Routine Description:

    Escapes a wide string for safe inclusion in JSON, handling special characters
    like quotes, backslashes, and control characters according to JSON spec.

Arguments:

    input - The wide string to escape for JSON inclusion.

Return Value:

    std::wstring containing the properly escaped JSON string.

--*/
inline
std::wstring
EscapeJsonStringW(
    _In_ const std::wstring& input
    )
{
    if (input.empty()) {
        return L"";
    }
    
    std::wstring output;
    output.reserve(input.length() * 2);
    
    for (wchar_t ch : input) {
        switch (ch) {
            case L'\"': output += L"\\\""; break;
            case L'\\': output += L"\\\\"; break;
            case L'\b': output += L"\\b"; break;
            case L'\f': output += L"\\f"; break;
            case L'\n': output += L"\\n"; break;
            case L'\r': output += L"\\r"; break;
            case L'\t': output += L"\\t"; break;
            default:
                if (ch < 32) {
                    wchar_t buffer[8];
                    swprintf_s(buffer, L"\\u%04x", static_cast<unsigned int>(ch));
                    output += buffer;
                } else {
                    output += ch;
                }
                break;
        }
    }
    
    return output;
}
