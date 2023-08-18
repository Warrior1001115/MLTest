#include "Neuron.h"
#include "net.h"
#include "trainingSet.h"

void showVectorVals(std::string label, std::vector<double>& v) {
  std::cout << label << " ";
  for (unsigned i = 0; i < v.size(); i++) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
}

int main() {
  srand(time(NULL));

  // read data from file
  trainingSet trainingData("testData.txt");

  // initialize net
  std::vector<unsigned> topology;
  trainingData.getTopology(topology);
  net simpleFC(topology);

  std::vector<double> inputVals, targetVals, resultVals;
  int trainingPass = 0;
  double epsilon = 0.00001;

  // save initial weights to compare
  auto initialWeights = simpleFC.getAllWeights();

  // train over all dataset once
  while (!trainingData.isEOF()) {
    ++trainingPass;
    std::cout << std::endl << "Pass: " << trainingPass << std::endl;

    if (trainingData.getNextInputs(inputVals) != topology[0]) {
      if (trainingData.isEOF()) break;
      std::cerr << "Invalid train sample detected!" << std::endl;
      continue;
    }

    // stochastic gradient descent (correct weights after every sample)
    //
    // input data
    showVectorVals("Input:", inputVals);
    simpleFC.feedForward(inputVals);

    // compare results with target
    trainingData.getTargetOutputs(targetVals);
    showVectorVals("Target:", targetVals);
    assert(targetVals.size() == topology.back());

    // show results
    simpleFC.getResults(resultVals);
    showVectorVals("Net result:", resultVals);

    // packpropagation
    simpleFC.backProp(targetVals);

    std::cout << "Error: " << simpleFC.getError() << std::endl
              << "Net average error: " << simpleFC.getRecentAverageError()
              << std::endl;

    // check convergence criterion
    if (simpleFC.getDeltaError() <= epsilon) {
      std::cout << std::endl
                << "Net was converged. Delta error: " << std::fixed
                << std::setprecision(7) << simpleFC.getDeltaError() << "."
                << std::endl;
      break;
    }
  }

  // print weights
  std::cout << std::endl
            << "Done" << std::endl
            << "Initial weights:" << std::endl;
  net::printWeights(initialWeights);

  std::cout << std::endl << "Trained weights:" << std::endl;
  net::printWeights(simpleFC.getAllWeights());

  // manual test
  std::string readline;
  while (std::getline(std::cin, readline)) {
    if (readline == "exit") break;

    std::vector<double> xTest;
    double x;
    std::stringstream ss(readline);
    while (ss >> x) xTest.push_back(x);

    if (simpleFC.getInDimension() != xTest.size()) {
      std::cout << "Wrong inner dimension! Need dim = "
                << simpleFC.getInDimension() << "." << std::endl;
      continue;
    }

    simpleFC.feedForward(xTest);
    simpleFC.getResults(resultVals);
    showVectorVals("Net result:", resultVals);
  }

#if defined(_MSC_VER) || defined(_WIN32)
  system("PAUSE");
#endif

  return (0);
}
