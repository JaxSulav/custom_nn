#pragma once

#include<iostream>
#include<math.h>

class Neuron
{
public:
    Neuron(double val);
    
private:
    double currentVal;
    double activatedVal;
    double derivativeVal;

public:
    void activation_fast_sigmoid();
    
    void derivation_fast_sigmoid();

    // Getters
    double getCurrentVal() {return this->currentVal;}
    double getActivatedVal() {return this->activatedVal;}
    double getDerivativeVal() {return this->derivativeVal;}

};
