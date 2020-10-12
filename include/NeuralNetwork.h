#pragma once

#include <iostream>
#include <vector>
#include "Layer.h"
#include "Matrix.h"


class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<int> topology);

public:
    void set_inputs_in_input_layer(std::vector<double> inputs);

    void print_layers_values();

private:
    // stores the values of number of neurons in each layer as index of the layers
    // If the nn contains 3 input layers, 2 hidden and 2 output layer,
    // topology vector contains (3,2,2)
    std::vector<int> topology;

    // Number of layers in neural network
    std::vector<Layer *> layers;

    // Number of weight matrices, size of this will be (size of topology - 1)
    std::vector<Matrix *> weightMatrices;

    std::vector<double> inputs;

};