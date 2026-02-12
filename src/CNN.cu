#include "../inc/CNN.h"
#include "../inc/Matrix.h"
#include <iostream>
#include <random>
#include <cmath>

CNN::CNN(const std::vector<int>& layerSizes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 1; i < layerSizes.size(); ++i) {
        Matrix weight(layerSizes[i], layerSizes[i - 1]);
        Matrix bias(layerSizes[i], 1);

        for (int r = 0; r < weight.getRows(); ++r) {
            for (int c = 0; c < weight.getCols(); ++c) {
                weight(r, c) = dist(gen);
            }
        }

        for (int r = 0; r < bias.getRows(); ++r) {
            bias(r, 0) = dist(gen);
        }

        weights.push_back(weight);
        biases.push_back(bias);
    }
}

void CNN::train(const std::vector<Matrix>& inputs, const std::vector<Matrix>& labels, int epochs, float learningRate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float totalLoss = 0.0f;
        for (size_t i = 0; i < inputs.size(); ++i) {
            Matrix output = forward(inputs[i]);
            backward(inputs[i], labels[i], learningRate);

            // Compute loss (mean squared error)
            for (int r = 0; r < output.getRows(); ++r) {
                for (int c = 0; c < output.getCols(); ++c) {
                    float diff = output(r, c) - labels[i](r, c);
                    totalLoss += diff * diff;
                }
            }
        }
        std::cout << "Epoch " << epoch + 1 << ", Loss: " << totalLoss / inputs.size() << std::endl;
    }
}

Matrix CNN::predict(const Matrix& input) {
    return forward(input);
}

Matrix CNN::forward(const Matrix& input) {
    // Validate input dimensions
    if (input.getRows() != weights[0].getCols() || input.getCols() != 1) {
        throw std::invalid_argument("Input dimensions do not match the expected dimensions for the first layer.");
    }

    Matrix activation = input;
    for (size_t i = 0; i < weights.size(); ++i) {
        Matrix z(activation.getRows(), weights[i].getCols());
        multiplyMatricesCUDA(activation, weights[i], z);
        for (int r = 0; r < z.getRows(); ++r) {
            for (int c = 0; c < z.getCols(); ++c) {
                z(r, c) += biases[i](r, 0);
                z(r, c) = 1.0f / (1.0f + std::exp(-z(r, c))); // Sigmoid activation
            }
        }
        activation = z;
    }
    return activation;
}

void CNN::backward(const Matrix& input, const Matrix& label, float learningRate) {
    std::vector<Matrix> activations;
    std::vector<Matrix> zs;

    // Forward pass to store activations and z values
    Matrix activation = input;
    activations.push_back(activation);
    for (size_t i = 0; i < weights.size(); ++i) {
        Matrix z(activation.getRows(), weights[i].getCols());
        multiplyMatricesCUDA(activation, weights[i], z);
        for (int r = 0; r < z.getRows(); ++r) {
            for (int c = 0; c < z.getCols(); ++c) {
                z(r, c) += biases[i](r, 0);
                z(r, c) = 1.0f / (1.0f + std::exp(-z(r, c))); // Sigmoid activation
            }
        }
        zs.push_back(z);
        activation = z;
        activations.push_back(activation);
    }

    // Backward pass
    std::vector<Matrix> deltas(weights.size());
    Matrix delta = activations.back();
    for (int r = 0; r < delta.getRows(); ++r) {
        for (int c = 0; c < delta.getCols(); ++c) {
            delta(r, c) = (delta(r, c) - label(r, c)) * activations.back()(r, c) * (1 - activations.back()(r, c));
        }
    }
    deltas.back() = delta;

    for (int l = weights.size() - 2; l >= 0; --l) {
        Matrix z = zs[l];
        Matrix nextDelta = deltas[l + 1];
        Matrix transposedWeight(weights[l + 1].getCols(), weights[l + 1].getRows());
        for (int r = 0; r < weights[l + 1].getRows(); ++r) {
            for (int c = 0; c < weights[l + 1].getCols(); ++c) {
                transposedWeight(c, r) = weights[l + 1](r, c);
            }
        }
        Matrix newDelta(z.getRows(), transposedWeight.getCols());
        multiplyMatricesCUDA(nextDelta, transposedWeight, newDelta);
        for (int r = 0; r < newDelta.getRows(); ++r) {
            for (int c = 0; c < newDelta.getCols(); ++c) {
                newDelta(r, c) *= z(r, c) * (1 - z(r, c));
            }
        }
        deltas[l] = newDelta;
    }

    // Update weights and biases
    for (size_t l = 0; l < weights.size(); ++l) {
        Matrix deltaWeight(weights[l].getRows(), weights[l].getCols());
        multiplyMatricesCUDA(activations[l], deltas[l], deltaWeight);
        for (int r = 0; r < weights[l].getRows(); ++r) {
            for (int c = 0; c < weights[l].getCols(); ++c) {
                weights[l](r, c) -= learningRate * deltaWeight(r, c);
            }
        }
        for (int r = 0; r < biases[l].getRows(); ++r) {
            biases[l](r, 0) -= learningRate * deltas[l](r, 0);
        }
    }
}