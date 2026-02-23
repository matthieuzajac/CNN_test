#include "Matrix.h"

Matrix transpose(const Matrix& mat) {
    Matrix result(mat.getCols(), mat.getRows());
    for (int i = 0; i < mat.getRows(); ++i) {
        for (int j = 0; j < mat.getCols(); ++j) {
            result(j, i) = mat(i, j);
        }
    }
    return result;
}