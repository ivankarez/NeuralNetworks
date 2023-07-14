# NeuralNetworks

A small and basic neural network library optimized for use with evolutionary algorithms.


## Basic usage
The _inputs_ and _outputs_ of the model are strictly float arrays. You can declare models by using the constructor of the `LayeredNetworkModel` type.

The first parameter is the size of the input array. After that you can add as many layers as you like. The size of the last layer will determine the size of the output array.

```C#
var activation = new LinearActivation();
var layer1 = new DenseLayer(3, activation, false);
var layer2 = new DenseLayer(2, activation, false);
var myModel = new LayeredNetworkModel(1, layer1, layer2);
```
---
You can run the model by calling the `Predict` function on it.
```C#
var result = myModel.Predict(new float[] { .1f });
```
**Important**: The model will re-use the same result array on the next run.

## Configuring
The `LayeredNetworkModel` has a `Parameters` property to store it's configuration. You can randomize the model by randomizing this collection.

You can also use this `Parameters` collection to save your model config or load it back.
```C#
for (int i = 0; i < myModel.Parameters.Count; i++)
{
    myModel.Parameters.Values[i] = /* Some random float */;
}
```