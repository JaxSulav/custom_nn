#pragma once

#include <iostream>
#include <vector>

class Matrix
{
public:
    Matrix(int rows, int cols, bool setRandom);

public:
    // Pointer to Matrix, transpose function
    Matrix *transpose();

    void print_matrix();

private:
    std::vector<std::vector<double>> matrixVals;

    int rows;
    int cols;

public:
    int getRows() {return this->rows;}
    int getCols() {return this->cols;}
};
