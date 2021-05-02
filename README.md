# Neural Network Builder

This application allows new neural network developers to play around with network architecture and to train/run neural network. Built from scratch, this beginning neural network utilizes a sigmoid activator function and stochastic gradient descent. The UI of the neural network allows users to customzie the architecture of a neural network without ever touching the code behind a neural work or attemping to omptimize layer usage and other critical features of a neural network that affect the effectiveness of a neural network.

<div align="center">
  <img src="https://user-images.githubusercontent.com/66758185/116801894-c089a680-aad3-11eb-84f5-88515da932ab.png" />
</div>

<div align="center">
  <video width="320" height="240" controls src="https://user-images.githubusercontent.com/66758185/116802098-c97b7780-aad5-11eb-95ff-b7681c81f2f2.mp4" />
</div>

## Dependencies 
- clang
- cmake
- cinder


### Windows 

#### Visual Studio
- Visual Studio 2017+

## Quick Start

- Download Cinder (latest version)
- Create a new project within the Cinder directory 

## Builder Controls

#### **Keyboard**

| Key      | Action |
| ----------- | ----------- |
| Right arrow      | Adds a layer to the end       |
| Left arrow   | Removes a layer from the end        |
| Up arrow      | Adds a neuron to the last layer |
| Down arrow      | Removes a neuron from the last layer |

#### **File Drops**
The file drop event is meant to receive a text file with training or predict data. You must drop Training Data in before you can use predict features!
- Key represents the text at the top of the text file


| Key      | Action |
| ----------- | ----------- |
| TRAIN      | Trains the neural network with the file data      |
| PREDICT | Predicts the data and displays the neural network output layer values in the outpout neruon | 

#### **Train file structure**
```
TRAIN
EXPECTED_VALUE_1 VALUE_1 VALUE_2 ...
EXPECTED VALUE_2 VALUE_1 VALUE_2 ...
.
.
.
```
#### **Train file structure**
```
PREDICT
VALUE_1 VALUE_2 ...
```
