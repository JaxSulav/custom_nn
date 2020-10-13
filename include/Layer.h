#pragma once

#include <iostream>
#include <vector>
#include "Neuron.h"
#include "Matrix.h"

extern int NEURON_CURRENT_VAL;
extern int NEURON_ACTIVATED_VAL;
extern int NEURON_DERIVATIVE_VAL;


class Layer
{
public:
    Layer(int size);

public:
    // convert neurons vector to 1D matrix for matrix multiplication in feed forward process
    Matrix *convert_to_1D_matrix(int neuronValType);

public:
    // Setters
    void set_neuron_val(int neuronIdx, double neuronVal) {this->neurons.at(neuronIdx)->setCurrentVal(neuronVal);}

    // Getters
    std::vector<Neuron *> get_neurons(){ return this->neurons;}

private:
    int size;
    std::vector<Neuron *> neurons;
    
};

