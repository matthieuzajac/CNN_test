#include "../inc/Matrix.h"
#include "../inc/CNN.h"
#include "../inc/DataLoader.h"
#include <iostream>

int main() {
    try {
        DataLoader trainLoader("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte");
        DataLoader testLoader("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte");

        std::vector<Matrix> trainImages = trainLoader.getImages();
        std::vector<Matrix> trainLabels = trainLoader.getLabels();
        std::vector<Matrix> testImages = testLoader.getImages();
        std::vector<Matrix> testLabels = testLoader.getLabels();

        CNN cnn({784, 128, 64, 10});
        cnn.train(trainImages, trainLabels, 10, 0.01f);

        int correct = 0;
        for (size_t i = 0; i < testImages.size(); ++i) {
            Matrix prediction = cnn.predict(testImages[i]);
            int predictedLabel = 0;
            float maxVal = prediction(0, 0);
            for (int j = 1; j < 10; ++j) {
                if (prediction(j, 0) > maxVal) {
                    maxVal = prediction(j, 0);
                    predictedLabel = j;
                }
            }

            int actualLabel = 0;
            for (int j = 0; j < 10; ++j) {
                if (testLabels[i](j, 0) == 1.0f) {
                    actualLabel = j;
                    break;
                }
            }

            if (predictedLabel == actualLabel) {
                ++correct;
            }
        }

        std::cout << "Accuracy: " << (static_cast<float>(correct) / testImages.size()) * 100.0f << "%" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}