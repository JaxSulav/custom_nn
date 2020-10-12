#include<iostream>
#include "../include/Neuron.h"

int main(int argc, char const *argv[])
{
    Neuron *n1 = new Neuron(0.9);
    std::cout << "val:" << n1->getCurrentVal() << std::endl;
    std::cout << "val1:" << n1->getActivatedVal() << std::endl;
    std::cout << "val2:" << n1->getDerivativeVal() << std::endl;



    return 0;
}
