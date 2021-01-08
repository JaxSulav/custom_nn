#include "../include/Utils.h"

std::string DEBUG_STRING = "false";
bool DEBUG = false;
int TEST_INT = 0;
std::string TEST_STRING = "";

// Parse the configurations from cfg file
int parse_config(std::string cfgFile)
{
    std::ifstream input;
    input.open(cfgFile);

    if (!input.is_open()){
        return 1;
    }

    int idx = 0;
    while(input){
        std::string line;
        std::getline(input, line, ':');

        input >> std::ws; // For Whitespaces

        if (idx == 0){
            input >> DEBUG_STRING;
            if (DEBUG_STRING == "true"){
                DEBUG = true;
            }
        }else if (idx == 1){
            input >> TEST_INT;
        }else if (idx == 2){
            input >> TEST_STRING;
        }
        else{
            break;
        }

        idx++;
    }

    return 0;
}


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
