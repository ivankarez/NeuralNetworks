using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class ConvolutionalLayer : IModelLayer
    {
        public int NodeCount { get; private set; }
        public int FilterSize { get; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        private float[] kernels;
        private float[] filter;

        public ConvolutionalLayer(int filterSize)
        {
            if (filterSize < 1) throw new ArgumentException("Filter size must be greater than 0", nameof(filterSize));

            FilterSize = filterSize;
            NodeCount = -1;
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(int inputSize)
        {
            NodeCount = inputSize - FilterSize + 1;
            if (NodeCount < 1) throw new ArgumentException("filterSize cannot be less than the size of the previous layer", nameof(inputSize));

            kernels = new float[NodeCount];
            filter = new float[FilterSize];

            State.Add("kernels", kernels);
            Parameters.Add("filter", filter);
        }

        public float[] Update(float[] inputValues)
        {
            for (int kernelIndex = 0; kernelIndex < NodeCount; kernelIndex++)
            {
                kernels[kernelIndex] = DotProductWithFilter(inputValues, kernelIndex);
            }

            return kernels;
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
