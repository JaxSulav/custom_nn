#include <iostream>
#include "../include/Neuron.h"
#include "../include/Utils.h"
#include "../include/Matrix.h"
#include "../include/NeuralNetwork.h"


int main(int argc, char const *argv[])
{
    parse_config("../config.cfg");
    if (DEBUG) {
        std::cout << TEST_INT << "\n";
        std::cout << TEST_STRING << "\n";
    }

    std::vector<int> topology;
    topology.push_back(3); // Input Layer   
    topology.push_back(5); // Hidden Layer
    topology.push_back(3); // Output Layer

    std::vector<double> inputs;
    inputs.push_back(1.0);
    inputs.push_back(0.0);
    inputs.push_back(1.0);

    NeuralNetwork *nn = new NeuralNetwork(topology);
    nn->set_inputs_in_input_layer(inputs);
    nn->setOutputTarget(inputs);


    for (int i=0; i<1; i++){
        nn->feed_forward();
        nn->calculate_MSE();
        nn->print_layers_values();
        std::cout << "Total error: " << nn->getTotalError() << std::endl;
        nn->back_propagation();
        std::cout << std::endl;
    }

    delete nn;

    return 0;
}
