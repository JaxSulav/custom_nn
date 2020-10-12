#include "../include/Layer.h"


Layer::Layer(int size) 
{
    this->size = size;

    for(size_t i=0; i<(size_t)size; i++){
        Neuron *n = new Neuron(0.00);
        this->neurons.push_back(n);
    }
}
