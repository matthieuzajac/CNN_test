#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

class Matrix {
private:
    int rows, cols;
    std::vector<float> data;

public:
    Matrix() : rows(0), cols(0) {}
    Matrix(int rows, int cols);
    float& operator()(int row, int col);
    const float& operator()(int row, int col) const;
    int getRows() const;
    int getCols() const;
    void randomize();
    void print() const;

    // Provide access to data for CUDA operations
    std::vector<float>& getData() {
        return data;
    }
    const std::vector<float>& getData() const {
        return data;
    }
};

void multiplyMatricesCUDA(const Matrix& A, const Matrix& B, Matrix& C);

#endif