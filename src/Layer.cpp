#include "../include/Layer.h"

int NEURON_CURRENT_VAL = 0;
int NEURON_ACTIVATED_VAL = 1;
int NEURON_DERIVATIVE_VAL = 2;

Layer::Layer(int size) 
{
    this->size = size;

    for(size_t i=0; i<(size_t)size; i++){
        Neuron *n = new Neuron(0.00);
        this->neurons.push_back(n);
    }
}


void Layer::set_neuron_val(int neuronIdx, double neuronVal) 
{
    this->neurons.at(neuronIdx)->setCurrentVal(neuronVal);
}


Matrix* Layer::convert_to_1D_matrix(int neuronValType) 
{
    // Create new matrix with 1 row and number of neurons in input layer as columns, with values 0
    Matrix *m = new Matrix(1, this->neurons.size(), false);

    // Fill the matrix with the curent value, activated value and derivative value for each neuron according to which is needed, governed by neuronValType
    for (int i=0; i<(int)this->neurons.size(); i++){
        if (neuronValType == NEURON_CURRENT_VAL){
            m->setValue(0, i, this->neurons.at(i)->getCurrentVal());
        }
        else if (neuronValType == NEURON_ACTIVATED_VAL){
            m->setValue(0, i, this->neurons.at(i)->getActivatedVal());
        }
        else if (neuronValType == NEURON_DERIVATIVE_VAL){
            m->setValue(0, i, this->neurons.at(i)->getDerivativeVal());
        }
        else {
            std::cout << "Unsupported argument in convert_to_1D_matrix()" << std::endl;
        }
    }

    return m;
}
