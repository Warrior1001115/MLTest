#Simple Fully Connected Neural Network on C++
This project realizes simple fully connected neural network.
This neural network train to evaluate NAND operation.
Neural network topology and training set generated by file __createData.cpp__ and saved to __testData.txt__. 
To create txt file use Developer command prompt for VS 2019: 
- cl createData.cpp
- createData.exe > testdata.txt

In this project used momentum optimizer and simple early stopping (when delta error become less than modifyed epsilon).
