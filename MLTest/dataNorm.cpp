// Normalization of test values stored in file and creating new file with normalized values.
// Do not include in project.

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>

double vectorMean(const std::vector<double>& values) {
  // calculate mean on given vector
  double sum = 0.0;
  for (const double& value : values) {
    sum += value;
  }
  return double(sum / values.size());
}

double standartDeviation(const std::vector<double>& values) {
  // calculate standart deviation on given vector
  double sum = 0.0;
  double mean = vectorMean(values);
  unsigned size = values.size();

  for (const double& value : values) {
    sum += std::pow(value - mean, 2.0);
  }

  return std::sqrt(sum / size);
}

int main() {
  std::ifstream readStream;
  std::ofstream writeStream;
  readStream.open("testDataExpFunc.txt");
  writeStream.open("testDataExpFuncNorm.txt");

  std::vector<double> inData;
  std::vector<double> targetData;

  std::string lineFromFile;
  std::string label;

  while (!readStream.eof()) {
    // read file line by line
    std::getline(readStream, lineFromFile);
    std::stringstream ss(lineFromFile);
    ss >> label;

    while (!ss.eof()) {
      // get values from line
      if (label == "topology:") break;
      if (label == "in:") {
        double val;

        while (ss >> val) inData.push_back(val);
        continue;
      }
      if (label == "out:") {
        double val;

        while (ss >> val) targetData.push_back(val);
        continue;
      }
    }
  }

  // Z-normalization

  std::vector<double> normTargetData;
  double mean = vectorMean(targetData);
  double stDev = standartDeviation(targetData);

  // normalization targetData
  for (const double& value : targetData) {
    normTargetData.push_back(value - mean / stDev);
  }

  // logarithmic transformation


  //std::cout << "----Initial data----" << std::endl;
  //for (const double& val : targetData) std::cout << val << " ";
  //std::cout << std::endl;

  //std::cout << "----Normalized data----" << std::endl;
  //for (const double& val : normTargetData) std::cout << val << " ";
  //std::cout << std::endl;

  // write in file
  for (unsigned i = 0; i < targetData.size(); i++) {
    writeStream << std::fixed << targetData[i] << " " << normTargetData[i]
                << std::endl;
  }

  std::cout << std::fixed << mean << std::endl << stDev;

  // for (const double &val : inData) std::cout << val << " ";
  // std::cout << std::endl;
  // for (const double &val : targetData) std::cout << val << " ";
  // std::cout << std::endl;
}