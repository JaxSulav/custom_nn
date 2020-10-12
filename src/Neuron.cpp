#include "../include/Neuron.h"

Neuron::Neuron(double val)
{
    this->currentVal = val;
    activation_fast_sigmoid();
    derivation_fast_sigmoid();
}

void Neuron::activation_fast_sigmoid() 
{
    // f(x) = x / (1 + |x|)
    this->activatedVal = this->currentVal / (1 + abs(this->currentVal));
}

void Neuron::derivation_fast_sigmoid() 
{
    // f'(x) = f(x) * (1 - f(x)) 
    this->derivativeVal = this->activatedVal * (1 - this->activatedVal);
}
