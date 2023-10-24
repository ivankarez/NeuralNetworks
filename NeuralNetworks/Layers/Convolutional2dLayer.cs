using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class Convolutional2dLayer : IModelLayer
    {
        public int NodeCount { get; private set; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }
        public Size2D InputSize { get; }
        public Size2D FilterSize { get; }
        public Stride2D Stride { get; }
        public bool UseBias { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer BiasInitializer { get; }

        private float[] nodeValues;
        private float[] biases;
        private float[,] filter;
        private int nodeValuesWidth;
        private int nodeValuesHeight;

        public Convolutional2dLayer(Size2D inputSize, Size2D filterSize, Stride2D stride,
            bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
        {
            if (inputSize == null) throw new ArgumentNullException(nameof(inputSize));
            if (filterSize == null) throw new ArgumentNullException(nameof(filterSize));
            if (stride == null) throw new ArgumentNullException(nameof(stride));
            if (kernelInitializer == null) throw new ArgumentNullException(nameof(kernelInitializer));
            if (biasInitializer == null) throw new ArgumentNullException(nameof(biasInitializer));

            if (filterSize.Width > inputSize.Width) throw new ArgumentException("Filter width cannot be greater than input width", nameof(filterSize.Width));
            if (filterSize.Height > inputSize.Height) throw new ArgumentException("Filter height cannot be greater than input height", nameof(filterSize.Height));

            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
            InputSize = inputSize;
            FilterSize = filterSize;
            Stride = stride;
            UseBias = useBias;
            KernelInitializer = kernelInitializer;
            BiasInitializer = biasInitializer;
        }

        public void Build(int inputSize)
        {
            var expectedInputSize = InputSize.Width * InputSize.Height;
            if (inputSize != expectedInputSize) throw new ArgumentException($"Input size must be {expectedInputSize}", nameof(inputSize));

            nodeValuesWidth = (InputSize.Width - FilterSize.Width) / Stride.Horizontal + 1;
            nodeValuesHeight = (InputSize.Height - FilterSize.Height) / Stride.Vertical + 1;
            NodeCount = nodeValuesWidth * nodeValuesHeight;
            nodeValues = new float[NodeCount];
            filter = KernelInitializer.GenerateValues2d(inputSize, NodeCount, FilterSize.Width, FilterSize.Height);
            biases = UseBias ? BiasInitializer.GenerateValues(inputSize, NodeCount, NodeCount) : new float[0];

            State.Add("nodeValues", nodeValues);
            Parameters.Add("filter", filter);
            Parameters.Add("biases", biases);
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeX = 0; nodeX < nodeValuesWidth; nodeX += 1)
            {
                for (int nodeY = 0; nodeY < nodeValuesHeight; nodeY += 1)
                {
                    var nodeValue = 0f;
                    for (int fx = 0; fx < filter.GetLength(0); fx += 1)
                    {
                        for (int fy = 0; fy < filter.GetLength(1); fy += 1)
                        {
                            var inputX = nodeX * Stride.Horizontal + fx;
                            var inputY = nodeY * Stride.Vertical + fy;
                            nodeValue += inputValues[inputX * InputSize.Width + inputY] * filter[fx, fy];
                        }
                    }
                    var nodeIndex = nodeX * nodeValuesHeight + nodeY;
                    if (UseBias)
                    {
                        nodeValue += biases[nodeIndex];
                    }
                    nodeValues[nodeIndex] = nodeValue;
                }
            }

            return nodeValues;
        }
    }
}
