#include "../include/NeuralNetwork.h"


NeuralNetwork::NeuralNetwork(std::vector<int> topology) 
{
    /* topology vector stores the values of number of neurons in each layer as index of the layers. If the nn contains 3 input layers, 2 hidden and 2 output layer, topology vector contains (3,2,2) */


    /* Create and push layers to layers vector */
    for (int i=0; i<topology.size(); i++){
        // Create a layer where number of neurons is the value at ith index of topology vector
        Layer *l = new Layer(topology.at(i));

        // Push the layer to the layers vector which stores the layers of our neural network
        this->layers.push_back(l);
    }


    /* Create and push weight matrices with random weights from normal distribution to weightMatrices vector */
    /* The size of weightMatrices will be (size of topology vec - 1) */
    for (int i=0; i<topology.size(); i++){
        // Create a weight matrix for 2 consecutive layers where the number of rows will be the number of neurons at first layer and number of columns will be the number of neurons at second layer. And the number of neurons in each layer is stored in topology vector
        Matrix *m = new Matrix(topology.at(i), topology.at(i+1), true);

        // Push the weight matrix to the weightMatrices vector 
        this->weightMatrices.push_back(m);
    }
}
