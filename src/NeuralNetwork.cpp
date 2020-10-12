#include "../include/NeuralNetwork.h"

// int NEURON_CURRENT_VAL = 0;
// int NEURON_ACTIVATED_VAL = 1;
// int NEURON_DERIVATIVE_VAL = 2;


NeuralNetwork::NeuralNetwork(std::vector<int> topology) 
{
    /* topology vector stores the values of number of neurons in each layer as index of the layers. If the nn contains 3 input layers, 2 hidden and 2 output layer, topology vector contains (3,2,2) */

    this->topology = topology;

    /* Create and push layers to layers vector */
    for (int i=0; i<(int)topology.size(); i++){
        // Create a layer where number of neurons is the value at ith index of topology vector
        Layer *l = new Layer(topology.at(i));

        // Push the layer to the layers vector which stores the layers of our neural network
        this->layers.push_back(l);
    }


    /* Create and push weight matrices with random weights from normal distribution to weightMatrices vector */
    /* The size of weightMatrices will be (size of topology vec - 1) */
    for (int i=0; i<(int)(topology.size()-1); i++){
        // Create a weight matrix for 2 consecutive layers where the number of rows will be the number of neurons at first layer and number of columns will be the number of neurons at second layer. And the number of neurons in each layer is stored in topology vector
        Matrix *m = new Matrix(topology.at(i), topology.at(i+1), true);

        // Push the weight matrix to the weightMatrices vector 
        this->weightMatrices.push_back(m);
    }
}


void NeuralNetwork::set_inputs_in_input_layer(std::vector<double> inputs) 
{
    this->inputs = inputs;

    // Set the value of each neuron in input layer(which is at the 0th index of layers vector) to corresponding values in inputs vector
    for (int i=0; i<(int)inputs.size(); i++){
        this->layers.at(0)->set_neuron_val(i, inputs.at(i));
    }
}


void NeuralNetwork::print_layers_values() 
{
    for (int i=0; i<(int)layers.size(); i++){
        std::cout << "LAYER: " << i << " :" << std::endl;
        // if (i == 0){
            Matrix *m = this->layers.at(i)->convert_to_1D_matrix(NEURON_CURRENT_VAL);
            m->print_matrix();
        // }
    }
}
