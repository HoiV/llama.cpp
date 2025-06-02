
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>

const size_t CHUNK_SIZE = 90 * 1024 * 1024; // 10 MB

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: splitter <file_path>" << std::endl;
        return 1;
    }

    std::string sourcePath = argv[1];
    std::ifstream sourceFile(sourcePath, std::ios::binary);
    if (!sourceFile) {
        std::cerr << "Failed to open source file." << std::endl;
        return 1;
    }

    std::filesystem::path pathObj(sourcePath);
    std::string baseName = pathObj.filename().string();
    std::string directory = pathObj.parent_path().string();
    if (!directory.empty() && directory.back() != '/' && directory.back() != '\\') {
        directory += std::filesystem::path::preferred_separator;
    }
    std::vector<char> buffer(CHUNK_SIZE);
    int index = 0;

    while (sourceFile) {
        sourceFile.read(buffer.data(), CHUNK_SIZE);
        std::streamsize bytesRead = sourceFile.gcount();
        if (bytesRead <= 0) break;

        char chunkName[256];
        snprintf(chunkName, sizeof(chunkName), "%s%s.%03d.part", directory.c_str(), baseName.c_str(), index);

        std::ofstream chunkFile(chunkName, std::ios::binary);
        if (!chunkFile) {
             std::cerr << "Failed to create chunk: " << chunkName << std::endl;
             return 1;
        }

        chunkFile.write(buffer.data(), bytesRead);
        std::cout << "Created chunk: " << chunkName << std::endl;
        ++index;
    }

    sourceFile.close();
    return 0;
}
