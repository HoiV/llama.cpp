#ifndef DTT_H
#define DTT_H
#include <vector>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <codecvt>
#include <locale>
#include "zip_file.h"

#pragma warning(disable:4996)// MSVC _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING);

inline std::vector<std::wstring> extractWTextInsideTags(const std::wstring& content, const std::wstring& tag)
{
    std::vector<std::wstring> results;
    std::wstring startTag = L"<" + tag;
    std::wstring endTag = L"</" + tag + L">";

    size_t pos = 0;
    while ((pos = content.find(startTag, pos)) != std::wstring::npos)
    {
        if (content[pos + startTag.size()] != L' ' && content[pos + startTag.size()] != L'>' && content[pos + startTag.size()] != L'/')
        {
            pos += startTag.size();
            continue;
        }
        size_t start = content.find(L'>', pos) + 1;
        size_t end = content.find(endTag, start);
        if (end == std::wstring::npos)
            break;

        std::wstring current = content.substr(start, end - start);
        if (!current.empty() && std::find(results.begin(), results.end(), current) == results.end())
            results.push_back(current);

        pos = end;
    }

    return results;
}

inline std::vector<std::string> extractTextInsideTags(const std::string& content, const std::string& tag)
{
    std::vector<std::string> results;
    std::string startTag = "<" + tag;
    std::string endTag = "</" + tag + ">";

    size_t pos = 0;
    while ((pos = content.find(startTag, pos)) != std::string::npos)
    {
        if (content[pos + startTag.size()] != ' ' && content[pos + startTag.size()] != '>' && content[pos + startTag.size()] != '/')
        {
            pos += startTag.size();
            continue;
        }
        size_t start = content.find('>', pos) + 1;
        size_t end = content.find(endTag, start);
        if (end == std::string::npos)
            break;

        std::string current = content.substr(start, end - start);
        if (!current.empty() && std::find(results.begin(), results.end(), current) == results.end())
            results.push_back(current);

        pos = end;
    }

    return results;
}

void flushToTextFile(std::string textFilename, std::string vdata) {
    std::ofstream writer(textFilename, std::ios::out | std::ios::trunc);
    if (!writer.is_open())
        return;
    std::locale locRS("sr_RS.UTF-8");
    writer.imbue(locRS);
    writer << vdata;
    writer.close();
}

inline std::string dtt(const std::filesystem::path& fp)
{
    std::string fnm = fp.stem().string();
    std::string tp = fp.parent_path().string();
    std::string pathTXT = tp + "\\" + fnm + ".txt";

    std::string vdata = "";

    try
    {
        zip_file archive(fp.string());
        if (!archive.has_file("word/document.xml")) {
            // no data to parse
            return vdata;
        }

        std::string data = archive.read("word/document.xml");

        #if 0 // Wstring

        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::wstring wdata = converter.from_bytes(data);
        std::vector<std::wstring> texts = extractWTextInsideTags(wdata, L"w:t");
        for (const auto& item : texts) {
            // vdata += item + L"\n";
            vdata += item + L" ";
        }

        std::wofstream writer(pathTXT, std::ios::out | std::ios::trunc);
        if (!writer.is_open())
            return vdata;
        std::locale locRS("sr_RS.UTF-8");
        writer.imbue(locRS);
        writer << vdata;
        writer.close();

        #else // char

        std::vector<std::string> texts = extractTextInsideTags(data, "w:t");        
        for (const auto& item : texts) {
            // vdata += item + "\n";
            vdata += item + " ";
        }

        #endif
    }
    catch (const std::exception& ex)
    {
        std::wcout << "Error occurred: " << ex.what() << std::endl;
        return vdata;
    }

    return vdata;
}


#endif // DTT_H


