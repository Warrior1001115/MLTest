#include "net.h"

#include <cassert>

#include "Neuron.h"

double net::smoothingFactor = 100.0;

net::net(const std::vector<unsigned>& topology) {
  recentAvgError = 0.0;
  error = 0.0;
  prevEpocError = 0.0;
  deltaError = 0.0;

  unsigned numLayers = topology.size();

  for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
    // add layer
    layers.push_back(Layer());
    // handling the case of the last layer
    unsigned numOutputs =
        layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

    for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
      // add neurons to last added layer
      layers.back().push_back(Neuron(numOutputs, neuronNum));
    }
    // add bias
    layers.back().back().setOutputVal(1.0);
  }
}

void net::feedForward(const std::vector<double>& inputVals) {
  assert(inputVals.size() == layers[0].size() - 1);

  // handling the case of the first layer
  for (unsigned i = 0; i < inputVals.size(); i++) {
    layers[0][i].setOutputVal(inputVals[i]);
  }

  for (unsigned layerNum = 1; layerNum < layers.size(); layerNum++) {
    Layer& prevLayer = layers[layerNum - 1];
    // layers[layerNum].size() - 1 is bias neuron,
    // so we don't update it, only its weights
    for (unsigned n = 0; n < layers[layerNum].size() - 1; n++) {
      layers[layerNum][n].feedForward(prevLayer);
    }
  }
}

void net::backProp(const std::vector<double>& targetVals) {
  Layer& outputLayer = layers.back();
  prevEpocError = error;

  // calculate RMSE
  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    double delta = targetVals[n] - outputLayer[n].getOutputVal();
    error += delta * delta;
  }
  error /= outputLayer.size() - 1;
  error = sqrt(error);

  recentAvgError =
      (recentAvgError * smoothingFactor + error) / (smoothingFactor + 1.0);

  // calculate gradients
  for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
    outputLayer[n].calcOutputGradient(targetVals[n]);
  }

  for (unsigned layerNum = layers.size() - 2; layerNum > 0; --layerNum) {
    Layer& hiddenLayer = layers[layerNum];
    Layer& nextLayer = layers[layerNum + 1];

    for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
      hiddenLayer[n].calcHiddenGradient(nextLayer);
    }

    // calculate delta error for early stop
    deltaError = abs(prevEpocError - error);
  }

  // update weights
  for (unsigned layerNum = layers.size() - 1; layerNum > 0; --layerNum) {
    Layer& layer = layers[layerNum];
    Layer& prevLayer = layers[layerNum - 1];

    for (unsigned n = 0; n < layer.size() - 1; ++n) {
      layer[n].updateInputWeights(prevLayer);
    }
  }
}

void net::getResults(std::vector<double>& resultVals) const {
  resultVals.clear();

  for (unsigned n = 0; n < layers.back().size() - 1; ++n) {
    resultVals.push_back(layers.back()[n].getOutputVal());
  }
}

void net::printWeights(
    const std::vector<std::vector<std::vector<double>>>& allWeights) {
  // parameters for formatting
  unsigned maxNumWeightSize = 0;
  unsigned maxNumNeuronsInLayer = 0;
  unsigned precisionValue = 8;
  unsigned cellWidth = precisionValue + 4;

  // calculate maxNumWeightSize and maxNumNeuronsInLayer for correct formatting
  for (const std::vector<std::vector<double>>& layer : allWeights) {
    if (layer.size() > maxNumNeuronsInLayer)
      maxNumNeuronsInLayer = layer.size();
    for (const std::vector<double>& neuron : layer) {
      if (neuron.size() > maxNumWeightSize) maxNumWeightSize = neuron.size();
    }
  }

  for (const std::vector<std::vector<double>>& layer : allWeights) {
    std::cout << "[";
    // default
    unsigned offset = 0;

    for (unsigned i = 0; i < layer.size() - 1; ++i) {
      // calculate front and back offset
      if (layer[i].size() < maxNumWeightSize) {
        offset = (unsigned)round(
            double((double(maxNumWeightSize) - layer[i].size()) * cellWidth) /
            2.0);
        std::cout << std::setw(offset) << "";
      }

      // print weight cells
      for (unsigned j = 0; j < layer[i].size(); ++j) {
        std::cout << std::fixed << std::showpos
                  << std::setprecision(precisionValue) << std::setw(cellWidth)
                  << layer[i][j];
      }

      std::cout << std::setw(offset) << "";
      std::cout << " |";
    }

    // calculate and print bias offset
    if (layer.size() < maxNumNeuronsInLayer) {
      unsigned biasCellsOffset = (cellWidth * maxNumWeightSize + 2) *
                                 (int(maxNumNeuronsInLayer) - layer.size());
      std::cout << std::setw(biasCellsOffset) << "";
    }

    std::cout << "|";
    std::cout << std::setw(offset) << "";

    // print bias cells
    for (unsigned j = 0; j < layer.back().size(); ++j) {
      std::cout << std::setw(cellWidth) << layer.back()[j];
    }

    std::cout << std::setw(offset) << "";
    std::cout << " ]" << std::endl;
  }
  std::cout << std::noshowpos;
}

// return vector[layer][neuron][weight]
std::vector<std::vector<std::vector<double>>> net::getAllWeights() const {
  std::vector<std::vector<std::vector<double>>> allWeights;
  for (const Layer& layer : layers) {
    std::vector<std::vector<double>> innerVector;
    for (const Neuron& neuron : layer) {
      innerVector.push_back(neuron.getWeights());
    }
    allWeights.push_back(innerVector);
  }
  return allWeights;
}
