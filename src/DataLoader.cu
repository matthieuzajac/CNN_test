#include "../inc/DataLoader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdint>

DataLoader::DataLoader(const std::string& imagePath, const std::string& labelPath) {
    loadImages(imagePath);
    loadLabels(labelPath);
}

void DataLoader::loadImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open image file: " + path);
    }

    int32_t magicNumber = 0;
    int32_t numImages = 0;
    int32_t numRows = 0;
    int32_t numCols = 0;

    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numImages), 4);
    file.read(reinterpret_cast<char*>(&numRows), 4);
    file.read(reinterpret_cast<char*>(&numCols), 4);

    magicNumber = __builtin_bswap32(magicNumber);
    numImages = __builtin_bswap32(numImages);
    numRows = __builtin_bswap32(numRows);
    numCols = __builtin_bswap32(numCols);

    for (int i = 0; i < numImages; ++i) {
        Matrix image(numRows, numCols);
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                uint8_t pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image(r, c) = static_cast<float>(pixel) / 255.0f;
            }
        }
        // Flatten the image into a vector (784x1 for MNIST)
        Matrix flattenedImage(numRows * numCols, 1);
        for (int r = 0; r < numRows; ++r) {
            for (int c = 0; c < numCols; ++c) {
                flattenedImage(r * numCols + c, 0) = image(r, c);
            }
        }
        images.push_back(flattenedImage);
    }
}

void DataLoader::loadLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open label file: " + path);
    }

    int32_t magicNumber = 0;
    int32_t numLabels = 0;

    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numLabels), 4);

    magicNumber = __builtin_bswap32(magicNumber);
    numLabels = __builtin_bswap32(numLabels);

    for (int i = 0; i < numLabels; ++i) {
        uint8_t label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        Matrix labelMatrix(10, 1);
        labelMatrix(static_cast<int>(label), 0) = 1.0f;
        labels.push_back(labelMatrix);
    }
}

std::vector<Matrix> DataLoader::getImages() const {
    return images;
}

std::vector<Matrix> DataLoader::getLabels() const {
    return labels;
}