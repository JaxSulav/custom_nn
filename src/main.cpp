#include<iostream>
#include "../include/Neuron.h"

int main(int argc, char const *argv[])
{
    Neuron *n = new Neuron(0.9);
    std::cout << "val:" << n->getCurrentVal() << std::endl;
    std::cout << "val1:" << n->getActivatedVal() << std::endl;
    std::cout << "val2:" << n->getDerivativeVal() << std::endl;



    return 0;
}
