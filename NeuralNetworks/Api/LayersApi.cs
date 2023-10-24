﻿using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;

namespace Ivankarez.NeuralNetworks.Api
{
    public class LayersApi
    {
        internal LayersApi() { }

        /// <summary>
        /// Creates and returns a DenseLayer, a fundamental building block for neural networks, with the specified configuration.
        /// </summary>
        /// <param name="nodeCount">The number of nodes (neurons) in the dense layer.</param>
        /// <param name="activation">The activation function to be used in the dense layer. Defaults to the Sigmoid activation.</param>
        /// <param name="useBias">A flag indicating whether bias terms should be used in the layer. Defaults to true.</param>
        /// <param name="kernelInitializer">The initializer for weights (kernels) of the layer. Defaults to Glorot Uniform initialization.</param>
        /// <param name="biasInitializer">The initializer for bias terms of the layer. Defaults to initializing with zeros.</param>
        /// <returns>A DenseLayer instance configured with the specified parameters.</returns>
        public DenseLayer Dense(int nodeCount, IActivation activation = null, bool useBias = true,
            IInitializer kernelInitializer = null, IInitializer biasInitializer = null)
        {
            activation ??= NN.Activations.Sigmoid();
            kernelInitializer ??= NN.Initializers.GlorotUniform();
            biasInitializer ??= NN.Initializers.Zeros();

            return new DenseLayer(nodeCount, activation, useBias, kernelInitializer, biasInitializer);
        }

        /// <summary>
        /// Creates and returns a Simple Recurrent Layer.
        /// </summary>
        /// <param name="nodeCount">The number of nodes (neurons) in the recurrent layer.</param>
        /// <param name="activation">The activation function to be used in the recurrent layer. Defaults to the Sigmoid activation.</param>
        /// <param name="useBias">A flag indicating whether bias terms should be used in the layer. Defaults to true.</param>
        /// <param name="kernelInitializer">The initializer for weights (kernels) of the layer. Defaults to Glorot Uniform initialization.</param>
        /// <param name="biasInitializer">The initializer for bias terms of the layer. Defaults to initializing with zeros.</param>
        /// <param name="recurrentInitializer">The initializer for recurrent weights. Defaults to Glorot Uniform initialization.</param>
        /// <returns>A Simple Recurrent Layer instance configured with the specified parameters.</returns>
        public RecurrentLayer SimpleRecurrent(int nodeCount, IActivation activation = null, bool useBias = true,
            IInitializer kernelInitializer = null, IInitializer biasInitializer = null, IInitializer recurrentInitializer = null)
        {
            activation ??= NN.Activations.Sigmoid();
            kernelInitializer ??= NN.Initializers.GlorotUniform();
            biasInitializer ??= NN.Initializers.Zeros();
            recurrentInitializer ??= NN.Initializers.GlorotUniform();

            return new RecurrentLayer(nodeCount, activation, useBias, kernelInitializer, biasInitializer, recurrentInitializer);
        }

        /// <summary>
        /// Creates and returns a 1D Convolutional Layer.
        /// </summary>
        /// <param name="filterSize">The size of the convolutional filters in the layer.</param>
        /// <param name="useBias">A flag indicating whether bias terms should be used in the layer. Defaults to true.</param>
        /// <param name="kernelInitializer">The initializer for the filter kernels of the layer. Defaults to Glorot Uniform initialization.</param>
        /// <param name="biasInitializer">The initializer for bias terms of the layer. Defaults to initializing with zeros.</param>
        /// <returns>A 1D Convolutional Layer instance configured with the specified parameters.</returns>
        public ConvolutionalLayer Conv1D(int filterSize, bool useBias = true, IInitializer kernelInitializer = null, IInitializer biasInitializer = null)
        {
            kernelInitializer ??= NN.Initializers.GlorotUniform();
            biasInitializer ??= NN.Initializers.Zeros();

            return new ConvolutionalLayer(filterSize, useBias, kernelInitializer, biasInitializer);
        }

        /// <summary>
        /// Creates and returns a 2D Convolutional Layer.
        /// </summary>
        /// <param name="inputWidth">The width of the input feature map to the layer.</param>
        /// <param name="inputHeight">The height of the input feature map to the layer.</param>
        /// <param name="filterWidth">The width of the convolutional filters in the layer.</param>
        /// <param name="filterHeight">The height of the convolutional filters in the layer.</param>
        /// <param name="strideX">The horizontal stride for filter movement. Defaults to 1.</param>
        /// <param name="strideY">The vertical stride for filter movement. Defaults to 1.</param>
        /// <param name="useBias">A flag indicating whether bias terms should be used in the layer. Defaults to true.</param>
        /// <param name="kernelInitializer">The initializer for the filter kernels of the layer. Defaults to Glorot Uniform initialization.</param>
        /// <param name="biasInitializer">The initializer for bias terms of the layer. Defaults to initializing with zeros.</param>
        /// <returns>A 2D Convolutional Layer instance configured with the specified parameters.</returns>
        public Convolutional2dLayer Conv2D(int inputWidth, int inputHeight, int filterWidth, int filterHeight,
            int strideX = 1, int strideY = 1, bool useBias = true, IInitializer kernelInitializer = null, IInitializer biasInitializer = null)
        {
            kernelInitializer ??= NN.Initializers.GlorotUniform();
            biasInitializer ??= NN.Initializers.Zeros();

            return new Convolutional2dLayer(inputWidth, inputHeight, filterWidth, filterHeight, strideX, strideY, useBias, kernelInitializer, biasInitializer);
        }

        /// <summary>
        /// Creates and returns a 1D Pooling Layer.
        /// </summary>
        /// <param name="window">The size of the pooling window, which determines the area over which pooling is applied.</param>
        /// <param name="stride">The stride for pooling operations. Defaults to 1.</param>
        /// <param name="poolingType">The type of pooling. Defaults to Max pooling.</param>
        /// <returns>A 1D Pooling Layer instance configured with the specified parameters.</returns>
        public PoolingLayer Pooling1D(int window, int stride = 1, PoolingType poolingType = PoolingType.Max)
        {
            return new PoolingLayer(window, stride, poolingType);
        }

        /// <summary>
        /// Creates and returns a 2D Pooling Layer for a neural network.
        /// </summary>
        /// <param name="inputSize">The size of the input feature map to the layer (width and height).</param>
        /// <param name="windowSize">The size of the pooling window, which determines the area over which pooling is applied (width and height).</param>
        /// <param name="stride">The horizontal and vertical stride for pooling operations. Defaults to (1, 1).</param>
        /// <param name="poolingType">The type of pooling. Defaults to Max pooling.</param>
        /// <returns>A 2D Pooling Layer instance configured with the specified parameters.</returns>
        public Pooling2dLayer Pooling2D(Size2D inputSize, Size2D windowSize, Stride2D stride = null, PoolingType poolingType = PoolingType.Max)
        {
            stride ??= new Stride2D(1, 1);

            return new Pooling2dLayer(inputSize, windowSize, stride, poolingType);
        }
    }
}
