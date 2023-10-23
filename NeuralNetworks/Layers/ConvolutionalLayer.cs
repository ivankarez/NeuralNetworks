using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class ConvolutionalLayer : IModelLayer
    {
        public int NodeCount { get; private set; }
        public int FilterSize { get; }
        public bool UseBias { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer BiasInitializer { get; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        private float[] nodeValues;
        private float[] filter;
        private float[] biases;

        public ConvolutionalLayer(int filterSize, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
        {
            if (filterSize < 1) throw new ArgumentException("Filter size must be greater than 0", nameof(filterSize));
            if (kernelInitializer == null) throw new ArgumentNullException(nameof(kernelInitializer));
            if (biasInitializer == null) throw new ArgumentNullException(nameof(biasInitializer));

            FilterSize = filterSize;
            UseBias = useBias;
            KernelInitializer = kernelInitializer;
            BiasInitializer = biasInitializer;
            NodeCount = -1;
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(int inputSize)
        {
            NodeCount = inputSize - FilterSize + 1;
            if (NodeCount < 1) throw new ArgumentException("filterSize cannot be less than the size of the previous layer", nameof(inputSize));

            nodeValues = new float[NodeCount];
            filter = KernelInitializer.GenerateValues(inputSize, NodeCount, FilterSize);
            biases = UseBias ? BiasInitializer.GenerateValues(NodeCount, NodeCount, NodeCount) : new float[0];

            State.Add("nodeValues", nodeValues);
            Parameters.Add("filter", filter);
            Parameters.Add("biases", biases);
        }

        public float[] Update(float[] inputValues)
        {
            for (int kernelIndex = 0; kernelIndex < NodeCount; kernelIndex++)
            {
                nodeValues[kernelIndex] = DotProductWithFilter(inputValues, kernelIndex);
                if (UseBias)
                {
                    nodeValues[kernelIndex] += biases[kernelIndex];
                }
            }

            return nodeValues;
        }

        private float DotProductWithFilter(float[] inputValue, int windowStart)
        {
            var sum = 0f;
            for (int i = 0; i < FilterSize; i++)
            {
                sum += inputValue[windowStart + i] * filter[i];
            }

            return sum;
        }
    }
}
