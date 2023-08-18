// Test data generating
// Do not include in project.
// 
// 
// To compile use Developer command prompt for VS:
// cl createData.cpp
// 
// And launch to write file:
// createData.exe > testData.txt

#include<iostream>
#include<cmath>
#include<cstdlib>
#include <iomanip>
#include <time.h>

// NAND dataset

int main() {
  std::cout << "topology: 2 3 1" << std::endl;
  for (int i = 2000; i >= 0; i--) {
    int first = (int)(2.0 * rand() / double(RAND_MAX));
    int second = (int)(2.0 * rand() / double(RAND_MAX));
    int result = !(first & second);

    std::cout << "in: " << first << ".0 " << second << ".0" << std::endl;
    std::cout << "out: " << result << ".0" << std::endl;
  }
}

// exponent function dataset (testing)

// int main() {
//  srand(time(NULL));
//  std::cout << "topology: 1 3 1" << std::endl;
//  for (int i = 0; i < 200; i++) {
//    double x = double(i) - 100.0;
//    double y = std::exp(double(x)) + (rand() / double(RAND_MAX) - 0.5);
//
//    std::cout << std::fixed << "in: " << std::setprecision(4) << x << std::endl;
//    std::cout << std::fixed << "out: " << std::setprecision(4) << y
//              << std::endl;  
//  }
//}