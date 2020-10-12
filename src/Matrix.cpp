#include "../include/Matrix.h"
#include "../include/Utils.h"


Matrix::Matrix(int rows, int cols, bool setRandom) 
{
    this->rows = rows;
    this->cols = cols;

    for (int i=0; i<rows; i++){
        std::vector<double> colVals;

        for (int j=0; j<cols; j++){
            double val = 0.00;

            if (setRandom){
                val = gen_rand_num_norm_dist();
            }

            colVals.push_back(val);
        }

        this->matrixVals.push_back(colVals);
    }
}


void Matrix::print_matrix() 
{
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            std::cout << this->matrixVals.at(i).at(j) << "\t";
        }
        std::cout << "\n";
    }
}
