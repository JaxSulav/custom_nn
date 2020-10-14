#include <iostream>
#include "../include/Neuron.h"
#include "../include/Utils.h"
#include "../include/Matrix.h"
#include "../include/NeuralNetwork.h"


int main(int argc, char const *argv[])
{
    // Neuron *n1 = new Neuron(0.9);
    // std::cout << "val:" << n1->getCurrentVal() << std::endl;
    // std::cout << "val1:" << n1->getActivatedVal() << std::endl;
    // std::cout << "val2:" << n1->getDerivativeVal() << std::endl;

    // std:: cout << gen_rand_num_norm_dist() << "\n" << std::endl;

    // Matrix *m = new Matrix(3, 2, true);
    // m->print_matrix();

    // std::cout << "-----Transpose-------" << std::endl;

    // Matrix *mT = m->transpose();
    // mT->print_matrix();

    std::vector<int> topology;
    topology.push_back(3); // Input Layer   
    topology.push_back(2); // Hidden Layer
    topology.push_back(3); // Output Layer

    std::vector<double> inputs;
    inputs.push_back(1.0);
    inputs.push_back(0.0);
    inputs.push_back(1.0);

    NeuralNetwork *nn = new NeuralNetwork(topology);
    nn->set_inputs_in_input_layer(inputs);
    nn->setOutputTarget(inputs);


    for (int i=0; i<100000; i++){
        nn->feed_forward();
        nn->calculate_MSE();
        // nn->print_layers_values();
        std::cout << "Total error: " << nn->getTotalError() << std::endl;
        nn->back_propagation();
        std::cout << std::endl;
    }


    return 0;
}
