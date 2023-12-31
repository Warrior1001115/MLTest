// Class to read data from file

#pragma once

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

class trainingSet {
 public:
  trainingSet(const std::string filename);
  bool isEOF() { return trainingDataFile.eof(); };
  void getTopology(std::vector<unsigned> &topology);
  unsigned getNextInputs(std::vector<double> &inputVals);
  unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

 private:
  unsigned getNext(const char *inOut, std::vector<double> & inputVals);
  std::ifstream trainingDataFile;
};
