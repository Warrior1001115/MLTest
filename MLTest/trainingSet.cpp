#include "trainingSet.h"

trainingSet::trainingSet(const std::string filename) {
  trainingDataFile.open(filename.c_str());
}

// for inner use
unsigned trainingSet::getNext(const char* inOut,
                              std::vector<double>& valToGet) {
  valToGet.clear();

  std::string line;
  std::getline(trainingDataFile, line);
  std::stringstream ss(line);

  std::string label;
  ss >> label;

  // compare readed line label with needed label
  if (label.compare(inOut) == 0) {
    double oneVal;

    while (ss >> oneVal) {
      valToGet.push_back(oneVal);
    }
  }

  return valToGet.size();
}

// almost like getNext()
void trainingSet::getTopology(std::vector<unsigned>& topology) {
  std::string line;
  std::string label;

  std::getline(trainingDataFile, line);
  std::stringstream ss(line);
  ss >> label;

  if (this->isEOF() || label.compare("topology:") != 0) {
    abort();
  }

  while (!ss.eof()) {
    unsigned n;
    ss >> n;
    topology.push_back(n);
  }
}

unsigned trainingSet::getNextInputs(std::vector<double>& inputVals) {
  return getNext("in:", inputVals);
}

unsigned trainingSet::getTargetOutputs(std::vector<double>& targetOutputVals) {
  return getNext("out:", targetOutputVals);
}
