#pragma once
#include <vector>
#include <cmath>

class Neuron;

typedef std::vector<Neuron> Layer;

struct Connection {
  double weight;
  double deltaWeight;
};

class Neuron {
 public:
  Neuron(unsigned numOutputs, unsigned myIndex);
  void setOutputVal(double val) { outputVal = val; }
  double getOutputVal() const { return outputVal; }
  double getActivation() const { return activation; }
  std::vector<double> getWeights() const;
  void feedForward(const Layer &prevLayer);
  void calcOutputGradient(double targetVal);
  void calcHiddenGradient(const Layer &nextLayer);	
  void updateInputWeights(Layer &prevLayer);

 private:
  static double lr;
  static double alpha;
  double outputVal;
  double activation;
  std::vector<Connection> outputWeights;
  unsigned myIndex;
  double gradient;
  static double randomWeight() { return rand() / double(RAND_MAX); } // range [-1.0 ... 1.0]
  static double activationFunction(double x);
  static double activationFunctionDerivative(double x);
  double sumGradWeight(const Layer &nextLayer) const;
};
