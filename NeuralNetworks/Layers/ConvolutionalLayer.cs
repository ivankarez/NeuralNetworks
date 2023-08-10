using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class ConvolutionalLayer : IModelLayer
    {
        public int NodeCount { get; private set; }
        public int FilterSize { get; }

        private ValueStoreRange kernels;
        private ValueStoreRange filter;

        public ConvolutionalLayer(int filterSize)
        {
            if (filterSize < 1) throw new ArgumentException("Filter size must be greater than 0", nameof(filterSize));

            FilterSize = filterSize;
            NodeCount = -1;
        }

        public void Build(int inputSize, ValueStore parameters, ValueStore state)
        {
            NodeCount = inputSize - FilterSize + 1;
            if (NodeCount < 1) throw new ArgumentException("filterSize cannot be less than the size of the previous layer", nameof(inputSize));

            kernels = state.AllocateRange(NodeCount);
            filter = parameters.AllocateRange(FilterSize);
        }

        public IValueArray Update(IValueArray inputValues)
        {
            for (int kernelIndex = 0; kernelIndex < NodeCount; kernelIndex++)
            {
                kernels[kernelIndex] = DotProductWithFilter(inputValues, kernelIndex);
            }

            return kernels;
        }

        private float DotProductWithFilter(IValueArray inputValue, int windowStart)
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
