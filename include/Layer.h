#pragma once

#include <iostream>
#include <vector>
#include "Neuron.h"


class Layer
{
public:
    Layer(int size);

private:
    int size;
    std::vector<Neuron *> neurons;
};

