#include "../include/Utils.h"


// Generate random number from normal distribution with a mean of 0 and standard deviation of 1
double gen_rand_num_norm_dist()
{
    auto mean = 0;
    auto stdDev = 1;

    std::random_device randDev;
    std::mt19937_64 randGenerator(randDev());
    std::normal_distribution<> distribution(mean, stdDev);

    double randVal = distribution(randGenerator);

    while (randVal < 0 || randVal > 1){
        randVal = distribution(randGenerator);
    }

    return randVal;
}


Matrix* multiply_matrices(Matrix *m1, Matrix *m2) 
{
    if (m1->getCols() != m2->getRows()){
        std::cout << "Matrix cannot be multiplied" << std::endl;
    }
    
    Matrix *m3 = new Matrix(m1->getRows(), m2->getCols(), false);

    for (int i=0; i<m1->getRows(); i++){
        for (int j=0; j<m2->getCols(); j++){
            for (int k=0; k<m2->getRows(); k++){
                double mul = m1->getValue(i, k) * m2->getValue(k, j);
                double resVal = m3->getValue(i, j) + mul;
                m3->setValue(i, j, resVal);
            }
        }
    }
    
    return m3;
}
