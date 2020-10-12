#include <iostream>
#include "../include/Neuron.h"
#include "../include/Utils.h"
#include "../include/Matrix.h"

int main(int argc, char const *argv[])
{
    Neuron *n1 = new Neuron(0.9);
    std::cout << "val:" << n1->getCurrentVal() << std::endl;
    std::cout << "val1:" << n1->getActivatedVal() << std::endl;
    std::cout << "val2:" << n1->getDerivativeVal() << std::endl;

    std:: cout << gen_rand_num_norm_dist() << "\n" << std::endl;

    Matrix *m = new Matrix(3, 3, true);
    m->print_matrix();

    return 0;
}
