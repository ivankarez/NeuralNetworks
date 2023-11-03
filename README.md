# NeuralNetworks
Introducing a lightweight and specialized neural network library for C#, meticulously designed to complement evolutionary algorithms. Our library is optimized for the crucial task of computing forward passes within neural networks, without the overhead of built-in backpropagation or conventional learning algorithms.

## Key Features:
- Efficiency and Memory Optimization: Our neural network library prioritizes resource efficiency, minimizing memory usage and allocations. It achieves this by utilizing C# float arrays for inputs and outputs, while intelligently reusing arrays to enhance performance across successive passes.
- Keras like API for a familiar interface.
- No extra dependencies. Makes it easy to drop it into your Unity or C# project as a single DLL file.

## Getting Started
### Install
You can install NeuralNetworks from [NuGet](https://www.nuget.org/packages/Ivankarez.NeuralNetworks/). For Unity projects, it's is recommended to install it using [NuGetForUnity](https://github.com/GlitchEnzo/NuGetForUnity).
```shell
dotnet add package Ivankarez.NeuralNetworks
```
If you prefer, you can also drop it into your project as a DLL file. You can download the latest build from the [current release](https://github.com/ivankarez/NeuralNetworks/releases/latest).
### Basic Usage
Creating a simple neural network with dense layers should be very similar if you are familiar with keras. You can reach all the functionality through the `NN` class from the `Ivankarez.NeuralNetworks.Api` namespace. This is the base class to declare your network. The NN api provides an easy way to access the capabilities of the library. You can use `NN.Layers` to create new layers, `NN.Activations` to access activation functions and so on. More of this in the [Features](#features) section.

Sample code to create layered model with 3 dense layers. The node count of the last layer will determine the output size of the network. In this example we create a network with this configuration:
 - An input of size 3
 - A dense layer with 10 neurons
 - A dense layer with 3 neurons, using the Tanh activation
 - A dense layer with 2 neurons. Since this is the last layer, it means this network will have an output size of 2.
```C#
using Ivankarez.NeuralNetworks.Api;

var neuralNetwork = NN.Models.Layered(NN.Size.Of(3),
        NN.Layers.Dense(10),
        NN.Layers.Dense(3, activation: NN.Activations.Tanh()),
        NN.Layers.Dense(2));

var result = neuralNetwork.FeedForward(new float[] {1, 2, 3});
Console.WriteLine($"Result: {string.Join(", ", result)}");
```

### Acessing paramters
To access parameters of a network (weights, biases etc...) you can iterate trough the layers of a model, and access it's parameters via the `Parameters` property.
```C#
var model = NN.Models.Layered(/*Any model config*/);
foreach(var layer in model.Layers) {
        var parameters = layer.Parameters;
}
```

If you want just a simple `float[]` of the parameters to store them (or used them as a DNA in a genetic algorithm), you can use the `GetParametersFlat` and `SetParametersFlat` extension methods of the model.
```C#
var model = NN.Models.Layered(/*Any model config*/);
var oldParameters = model.GetParametersFlat();
var newParameters = /* New parameters as a float array */;
model.SetParametersFlat(newParameters);
```

If you just want to count the number of parameters, you can use the `CountParameters()` extension method of the model.

### Demo Programs
There is a [NeuralNetworks.Demos](https://github.com/ivankarez/NeuralNetworks.Demos) repository, where we plan to collect different demos for this library. Currently it only contains a simple C# application where we train a neural network to learn classifying images of 'A' and 'B' characters.

## Features
This is a simple list of available features of this package. If you look for available parameters or default values, you can take a look at the corresponding API codes linked in the section headers.

### [Models](https://github.com/ivankarez/NeuralNetworks/blob/main/NeuralNetworks/Api/ModelsApi.cs)
- Layered Model

### [Layers](https://github.com/ivankarez/NeuralNetworks/blob/main/NeuralNetworks/Api/LayersApi.cs)
- Dense Layer
- Simple Recurrent Layer
- 1 dim convolution layer
- 2 dim convolution layer
- 1 dim pooling layer (min, max, average, sum)
- 2 dim pooling layer (min, max, average, sum)

### [Activations](https://github.com/ivankarez/NeuralNetworks/blob/main/NeuralNetworks/Api/ActivationsApi.cs)
- Linear
- Clamped linear
- Tanh
- Sigmoid
- Relu
- Leaky Relu

### [Initializers](https://github.com/ivankarez/NeuralNetworks/blob/main/NeuralNetworks/Api/InitializersApi.cs)
- Zeros
- Constant
- Uniform
- Normal
- Glorot uniform
- Glorot normal

### [Size](https://github.com/ivankarez/NeuralNetworks/blob/main/NeuralNetworks/Api/SizeApi.cs)
- *Of(int)*: Create a Size1D object
- *Of(int, int)*: Create a Size2D object

## Contributions
All contributions are welcome. For a starting point it's quite easy to implement other activation functions and initializers. Also extending test coverage, or simplify tests can be a good starting point.
