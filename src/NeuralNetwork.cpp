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


void NeuralNetwork::feed_forward() 
{
    /* Loop through each layer, get the neuron matrix for each layer and weights matrix and multiply these matrices. Also, set values for neurons in next layer */

    // We only loop till the (layers.size() -1) because the output neurons doesnot have weights

    for (int i=0; i<(int)(this->layers.size()-1); i++){
        Matrix *m1;

        // If the layer is input layer, get neuron matrix, else get the activated values of neuron matrix
        if (i == 0){
            m1 = this->getNeuronMatrix(i);
        }else{
            m1 = this->getActivatedNeuronMatrix(i);
        }

        // Get weight matrix and multiply m1 and m2
        Matrix *m2 = this->getWeightMatrix(i);
        Matrix *m3 = multiply_matrices(m1, m2);

        // Loop through the columns of multiplied matrix to set the values of neurons in the next layer
        for (int x=0; x<m3->getCols(); x++){
            this->set_each_neuron_value(i+1, x, m3->getValue(0, x));
        }
    }
}


void NeuralNetwork::print_layers_values() 
{
    for (int i=0; i<(int)layers.size(); i++){
        std::cout << "LAYER: " << i << " :" << std::endl;
        if (i == 0){
            Matrix *m = this->layers.at(i)->convert_to_1D_matrix(NEURON_CURRENT_VAL);
            m->print_matrix();
        }else{
            Matrix *m = this->layers.at(i)->convert_to_1D_matrix(NEURON_ACTIVATED_VAL);
            m->print_matrix();
        }

        if (i < (int)this->layers.size()-1){
            std::cout << "Weights Matrix: " << std::endl;
            this->getWeightMatrix(i)->print_matrix();
        }
        std::cout << "--------------------------" << std::endl;

    }
}


void NeuralNetwork::calculate_MSE() 
{
    if (this->target.size()==0){
        std::cout << "No output targets" << std::endl;
    }

    // The size of the target vector should be equal to the number of nurons in output layer
    if (this->target.size() != this->layers.at(this->layers.size()-1)->get_neurons().size()){
        std::cout << "Target size doesnot match output size" << std::endl;
    }

    int outputLayerIdx = this->layers.size()-1;
    std::vector<Neuron *> outputNeurons = this->layers.at(outputLayerIdx)->get_neurons();

    // calculate error and push to errors vector
    for (int i=0; i<(int)target.size(); i++){
        double tempErr = (outputNeurons.at(i)->getActivatedVal() - target.at(i));
        errors.push_back(tempErr);
        this->totalError += pow(tempErr, 2);
    }

    this->totalError = 0.5 * this->totalError;
    
    savedErrors.push_back(this->totalError);

}


void NeuralNetwork::back_propagation() 
{
    std::vector<Matrix *> newWeightsAll;
    Matrix *gradient;

    // Going from op to last hidden layer
    int outputLayerIdx = this->layers.size()-1;
    Matrix *derivativeValOfOutputNeurons = this->layers.at(outputLayerIdx)->convert_to_1D_matrix(NEURON_DERIVATIVE_VAL);

    Matrix *gradientsOp = new Matrix (1, this->layers.at(outputLayerIdx)->get_neurons().size(), false);

    for (int i=0; i< (int)this->layers.at(outputLayerIdx)->get_neurons().size(); i++){
        double derivativeVal = derivativeValOfOutputNeurons->getValue(0, i);
        double error = this->errors.at(i);
        double gradient = derivativeVal * error;
        gradientsOp->setValue(0, i, gradient);
    }

    int lastHiddenLayerIdx = outputLayerIdx - 1;
    Layer *LastHiddenLayer = this->layers.at(lastHiddenLayerIdx);
    Matrix *oldWeightsMatrixOp = this->weightMatrices.at(lastHiddenLayerIdx);
    Matrix *deltaWeightMatrixOp = multiply_matrices(gradientsOp->transpose(), LastHiddenLayer->convert_to_1D_matrix(NEURON_DERIVATIVE_VAL))->transpose();

    Matrix *updatedWeightsMatrixOp = new Matrix(deltaWeightMatrixOp->getRows(), deltaWeightMatrixOp->getCols(), false);

    for (int i=0; i<deltaWeightMatrixOp->getRows(); i++){
        for (int j=0; j<deltaWeightMatrixOp->getCols(); j++){
            double oldWeight = oldWeightsMatrixOp->getValue(i, j);
            double deltaWeight = deltaWeightMatrixOp->getValue(i, j);
            double newWeight =  oldWeight - deltaWeight;

            updatedWeightsMatrixOp->setValue(i, j, newWeight);
        }
    }

    newWeightsAll.push_back(updatedWeightsMatrixOp);

    // Get the gradients to the right
    gradient = new Matrix (gradientsOp->getRows(), gradientsOp->getCols(), false);

    for (int i=0; i<gradientsOp->getRows(); i++){
        for (int j=0; j<gradientsOp->getCols(); j++){
            gradient->setValue(i, j, gradientsOp->getValue(i, j));
        } 
    }


    // From hidden to input
    for (int i=lastHiddenLayerIdx; i>0; i--){
        Layer *l = this->layers.at(i);
        Matrix *derivativeValOfHiddenNeurons = l->convert_to_1D_matrix(NEURON_DERIVATIVE_VAL);
        Matrix *gradientsHidden = new Matrix (1, l->get_neurons().size(), false);
        Matrix *activatedValOfHiddenNeurons = l->convert_to_1D_matrix(NEURON_ACTIVATED_VAL);

        Matrix *weightMatrix = this->weightMatrices.at(i);
        Matrix *oldWeightsMatrixHidden = this->weightMatrices.at(i-1);

        for (int i=0; i<weightMatrix->getRows(); i++){
            double sum = 0;
            for (int j=0; j<weightMatrix->getCols(); j++){
                double product = gradient->getValue(0, j) * weightMatrix->getValue(i, j);
                
                sum += product;
            }

            double gradientVal = sum * activatedValOfHiddenNeurons->getValue(0, i);

            gradientsHidden->setValue(0, i, gradientVal);
        }

        Matrix *leftNeuronsMatrix = (i-1) == 0 ? this->layers.at(0)->convert_to_1D_matrix(NEURON_CURRENT_VAL) : this->layers.at(i-1)->convert_to_1D_matrix(NEURON_ACTIVATED_VAL);

        Matrix *deltaWeightsMatrixHidden = multiply_matrices(gradientsHidden->transpose(), leftNeuronsMatrix)->transpose();


        Matrix *updatedWeightsMatrixHidden = new Matrix (deltaWeightsMatrixHidden->getRows(), deltaWeightsMatrixHidden->getCols(), false);

        for (int i=0; i<updatedWeightsMatrixHidden->getRows(); i++){
            for (int j=0; j<updatedWeightsMatrixHidden->getCols(); j++){
                double oldWeightH = oldWeightsMatrixHidden->getValue(i, j);
                double deltaWeightH = deltaWeightsMatrixHidden->getValue(i, j);
                double newWeightH = oldWeightH - deltaWeightH;

                updatedWeightsMatrixHidden->setValue(i, j, newWeightH);
            } 
        }

        gradient = new Matrix (gradientsHidden->getRows(), gradientsHidden->getCols(), false);

        for (int i=0; i<gradientsHidden->getRows(); i++){
            for (int j=0; j<gradientsHidden->getCols(); j++){
                gradient->setValue(i, j, gradientsHidden->getValue(i, j));
            } 
        }

        newWeightsAll.push_back(updatedWeightsMatrixHidden);
    }

    std::reverse (newWeightsAll.begin(), newWeightsAll.end());

    this->weightMatrices = newWeightsAll; 
    
    std::cout << "Backpropagation finished" << std::endl;
}   
