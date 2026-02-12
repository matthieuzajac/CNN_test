#ifndef DATALOADER_H
#define DATALOADER_H

#include "Matrix.h"
#include <string>
#include <vector>

class DataLoader {
public:
    DataLoader(const std::string& imagePath, const std::string& labelPath);

    std::vector<Matrix> getImages() const;
    std::vector<Matrix> getLabels() const;

private:
    std::vector<Matrix> images;
    std::vector<Matrix> labels;

    void loadImages(const std::string& path);
    void loadLabels(const std::string& path);
};

#endif