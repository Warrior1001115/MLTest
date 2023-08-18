#include "Neuron.h"

double Neuron::lr = 0.15;   // learning rate
double Neuron::alpha = 0.3;  // momentum

Neuron::Neuron(unsigned numOutputs, unsigned myInd) {
  for (unsigned i = 0; i < numOutputs; ++i) {
    outputWeights.push_back(Connection());
    outputWeights.back().weight = randomWeight();  // initializing start weights
    outputWeights.back().deltaWeight = 0.0;
  }

  myIndex = myInd;
}

std::vector<double> Neuron::getWeights() const {
  std::vector<double> weights;
  for (const Connection &OneWeight : outputWeights) {
    weights.push_back(OneWeight.weight);
  }
  return weights;
}

void Neuron::feedForward(const Layer& prevLayer) {
  double sum = 0.0;
  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    sum += prevLayer[n].getOutputVal() *
           prevLayer[n].outputWeights[myIndex].weight;
  }

  activation = sum;
  outputVal = Neuron::activationFunction(sum);
}

double Neuron::activationFunction(double x) {
  // output range [-1.0 ... 1.0]
  return tanh(x);
}

double Neuron::activationFunctionDerivative(double x) {
  // derivative th(x)' = 1 - th(x)^2
  return 1.0 - std::pow(tanh(x), 2);
}

// gradient of the last layer neuron
// f'(a[l])*grad(z[l]) , where grad(z[l]) = predY - targY
void Neuron::calcOutputGradient(double targetVal) {
  double delta = outputVal - targetVal;
  gradient = delta * Neuron::activationFunctionDerivative(activation);
}

// gradient of the hidden layer neuron
// sum w[i]*z[i] * f'(a[l])
void Neuron::calcHiddenGradient(const Layer& nextLayer) {
  double dow = sumGradWeight(nextLayer);
  gradient = dow * Neuron::activationFunctionDerivative(activation);
}

// sum of all output connection weights 
double Neuron::sumGradWeight(const Layer& nextLayer) const {
  double sum = 0.0;

  for (unsigned n = 0; n < nextLayer.size() - 1; ++n) {
    sum += outputWeights[n].weight * nextLayer[n].gradient;
  }

  return sum;
}

void Neuron::updateInputWeights(Layer& prevLayer) {
  for (unsigned n = 0; n < prevLayer.size(); ++n) {
    Neuron& neuron = prevLayer[n];
    double oldDeltaWeight = neuron.outputWeights[myIndex].deltaWeight;

    double gradientOnWeights = neuron.getOutputVal() * gradient;

    // gradient descent step
    // w[l] = learningRate*actFunc(z[l])*gradient + momentum*oldDelta + w[l - 1] , where l is num layer
    double newDeltaWeight = lr * gradientOnWeights + alpha * oldDeltaWeight;

    neuron.outputWeights[myIndex].deltaWeight = newDeltaWeight;
    neuron.outputWeights[myIndex].weight -= newDeltaWeight;
  }
}