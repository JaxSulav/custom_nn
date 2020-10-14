#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "Layer.h"
#include "Matrix.h"
#include "Utils.h"


class NeuralNetwork
{
public:
    NeuralNetwork(std::vector<int> topology);

public:
    void set_inputs_in_input_layer(std::vector<double> inputs);

    void feed_forward();

    void print_layers_values();

    void calculate_errors();

    void calculate_MSE();

    void back_propagation();

public:
    //Setters
    void set_each_neuron_value(int layerIdx, int neuronIdx, double val) {this->layers.at(layerIdx)->set_neuron_val(neuronIdx, val);}
    void setOutputTarget(std::vector<double> target) {this->target = target;}

    // Getters
    Matrix *getNeuronMatrix(int index) {return this->layers.at(index)->convert_to_1D_matrix(NEURON_CURRENT_VAL);}
    Matrix *getActivatedNeuronMatrix(int index) {return this->layers.at(index)->convert_to_1D_matrix(NEURON_ACTIVATED_VAL);}
    Matrix *getDerivativeNeuronMatrix(int index) {return this->layers.at(index)->convert_to_1D_matrix(NEURON_DERIVATIVE_VAL);}
    Matrix *getWeightMatrix(int index) {return this->weightMatrices.at(index);}

    double getTotalError() {return this->totalError;}
    std::vector<double> getErrors() {return this->errors;}

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

    std::vector<double> target;
    double totalError;
    std::vector<double> errors;
    std::vector<double> savedErrors;

    std::vector<Matrix *> gradientMatrices;


};