using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class ConvolutionalLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; }
        public int FilterSize { get; }
        public int Stride { get; }
        public bool UseBias { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer BiasInitializer { get; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        private float[] nodeValues;
        private float[] filter;
        private float[] biases;

        public ConvolutionalLayer(int filterSize, int stride, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
        {
            if (filterSize < 1) throw new ArgumentException("Filter size must be greater than 0", nameof(filterSize));
            if (stride < 1) throw new ArgumentException("Stride must be greater than 0", nameof(stride));
            FilterSize = filterSize;
            Stride = stride;
            UseBias = useBias;
            KernelInitializer = kernelInitializer ?? throw new ArgumentNullException(nameof(kernelInitializer));
            BiasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(ISize inputSize)
        {
            if (FilterSize > inputSize.TotalSize) throw new ArgumentException("filterSize cannot be more than the size of the previous layer", nameof(inputSize));
            OutputSize = new Size1D(ConvolutionUtils.CalculateOutputSize(inputSize.TotalSize, FilterSize, Stride));

            nodeValues = new float[OutputSize.TotalSize];
            filter = KernelInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, FilterSize);
            biases = UseBias ? BiasInitializer.GenerateValues(OutputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize) : new float[0];

            State.Add("nodeValues", nodeValues);
            Parameters.Add("filter", filter);
            Parameters.Add("biases", biases);
        }

        public float[] Update(float[] inputValues)
        {
            for (int kernelIndex = 0; kernelIndex < OutputSize.TotalSize; kernelIndex++)
            {
                var value = DotProductWithFilter(inputValues, kernelIndex * Stride);
                if (UseBias)
                {
                    value += biases[kernelIndex];
                }
                nodeValues[kernelIndex] = value;
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
