using Ivankarez.NeuralNetworks.Abstractions;
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
        /// <param name="stride">The stride for the filter movement. Defaults to 1.</param>
        /// <param name="useBias">A flag indicating whether bias terms should be used in the layer. Defaults to true.</param>
        /// <param name="kernelInitializer">The initializer for the filter kernels of the layer. Defaults to Glorot Uniform initialization.</param>
        /// <param name="biasInitializer">The initializer for bias terms of the layer. Defaults to initializing with zeros.</param>
        /// <returns>A 1D Convolutional Layer instance configured with the specified parameters.</returns>
        public ConvolutionalLayer Conv1D(int filterSize, int stride = 1, bool useBias = true, IInitializer kernelInitializer = null, IInitializer biasInitializer = null)
        {
            kernelInitializer ??= NN.Initializers.GlorotUniform();
            biasInitializer ??= NN.Initializers.Zeros();

            return new ConvolutionalLayer(filterSize, stride, useBias, kernelInitializer, biasInitializer);
        }

        /// <summary>
        /// Creates and returns a 2D Convolutional Layer for a neural network.
        /// </summary>
        /// <param name="filterSize">The size of the convolutional filters in the layer (width and height).</param>
        /// <param name="stride">The horizontal and vertical stride for filter movement. Defaults to (1, 1).</param>
        /// <param name="useBias">A flag indicating whether bias terms should be used in the layer. Defaults to true.</param>
        /// <param name="kernelInitializer">The initializer for the filter kernels of the layer. Defaults to Glorot Uniform initialization.</param>
        /// <param name="biasInitializer">The initializer for bias terms of the layer. Defaults to initializing with zeros.</param>
        /// <returns>A 2D Convolutional Layer instance configured with the specified parameters.</returns>
        public Convolutional2dLayer Conv2D(Size2D filterSize, Stride2D stride = null,
            bool useBias = true, IInitializer kernelInitializer = null, IInitializer biasInitializer = null)
        {
            kernelInitializer ??= NN.Initializers.GlorotUniform();
            biasInitializer ??= NN.Initializers.Zeros();
            stride ??= new Stride2D(1, 1);

            return new Convolutional2dLayer(filterSize, stride, useBias, kernelInitializer, biasInitializer);
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
        /// <param name="windowSize">The size of the pooling window, which determines the area over which pooling is applied (width and height).</param>
        /// <param name="stride">The horizontal and vertical stride for pooling operations. Defaults to (1, 1).</param>
        /// <param name="poolingType">The type of pooling. Defaults to Max pooling.</param>
        /// <returns>A 2D Pooling Layer instance configured with the specified parameters.</returns>
        public Pooling2dLayer Pooling2D(Size2D windowSize, Stride2D stride = null, PoolingType poolingType = PoolingType.Max)
        {
            stride ??= new Stride2D(1, 1);

            return new Pooling2dLayer(windowSize, stride, poolingType);
        }

        /// <summary>
        /// Creates and returns a Gated Recurrent Unit (GRU) Layer configured with the specified parameters.
        /// </summary>
        /// <param name="nodeCount">The number of nodes (neurons) in the GRU layer.</param>
        /// <param name="activation">The activation function applied to the output of the recurrent units. Defaults to the hyperbolic tangent (Tanh) activation.</param>
        /// <param name="recurrentActivation">The activation function applied to the recurrent step. Defaults to the Sigmoid activation.</param>
        /// <param name="useBias">A flag indicating whether bias terms should be used in the GRU layer. Defaults to true.</param>
        /// <param name="kernelInitializer">The initializer for weights (kernels) of the GRU layer. Defaults to Glorot Uniform initialization.</param>
        /// <param name="recurrentInitializer">The initializer for recurrent weights of the GRU layer. Defaults to Glorot Uniform initialization.</param>
        /// <param name="biasInitializer">The initializer for bias terms of the GRU layer. Defaults to initializing with zeros.</param>
        /// <returns>A Gated Recurrent Unit (GRU) Layer instance configured with the specified parameters.</returns>
        public GruLayer GRU(Size1D nodeCount, IActivation activation = null, IActivation recurrentActivation = null, bool useBias = true,
            IInitializer kernelInitializer = null, IInitializer recurrentInitializer = null, IInitializer biasInitializer = null)
        {
            activation ??= NN.Activations.Tanh();
            recurrentActivation ??= NN.Activations.Sigmoid();
            kernelInitializer ??= NN.Initializers.GlorotUniform();
            recurrentInitializer ??= NN.Initializers.GlorotUniform();
            biasInitializer ??= NN.Initializers.Zeros();

            return new GruLayer(nodeCount, activation, recurrentActivation, useBias, kernelInitializer, recurrentInitializer, biasInitializer);
        }

        /// <summary>
        /// Creates and returns a Dropout Layer with the specified dropout rate.
        /// </summary>
        /// <param name="dropoutRate">The rate at which to drop out nodes in the layer.</param>"
        /// <param name="randomProvider">The random provider to use for generating random values. Defaults to NN.Random.Default().</param>
        /// <returns>A Dropout Layer instance configured with the specified parameters.</returns>
        public DropoutLayer Dropout(float dropoutRate, IRandomProvider randomProvider = null)
        {
            randomProvider ??= NN.Random.Default();

            return new DropoutLayer(dropoutRate, randomProvider);
        }
    }
}
