#include <iostream>
#include <iomanip>
#include <vector>

#include "Neuron.h"
#pragma once

class net {
 public:
  net(const std::vector<unsigned> &topology);
  void feedForward(const std::vector<double> &inputVals);
  void backProp(const std::vector<double> &targetVals);
  int getInDimension() const { return layers[0].size() - 1; }
  void getResults(std::vector<double> &resultVals) const;
  double getRecentAverageError() const { return recentAvgError; }
  double getDeltaError() const { return deltaError; }
  double getError() const { return error; }
  static void printWeights(const std::vector<std::vector<std::vector<double>>> &allWeights);
  std::vector<std::vector<std::vector<double>>> getAllWeights() const;
 private:
  std::vector<Layer> layers;  // layers[layerNumber][neuronNumber]
  double recentAvgError;
  double prevEpocError;
  double error;
  double deltaError;
  static double smoothingFactor;
};
