#ifndef CNN_H
#define CNN_H

#include "Matrix.h"
#include <vector>
#include <string>

class CNN {
public:
    CNN(const std::vector<int>& layerSizes);

    void train(const std::vector<Matrix>& inputs, const std::vector<Matrix>& labels, int epochs, float learningRate);
    Matrix predict(const Matrix& input);

private:
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;

    Matrix forward(const Matrix& input);
    void backward(const Matrix& input, const Matrix& label, float learningRate);
};

#endif