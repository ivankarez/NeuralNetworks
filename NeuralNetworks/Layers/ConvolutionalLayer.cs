using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
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
        public float[] Filter { get; set; }
        public float[] Biases { get; set; }
        public float[] Output { get; private set; }

        public ConvolutionalLayer(int filterSize, int stride, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
        {
            if (filterSize < 1) throw new ArgumentException("Filter size must be greater than 0", nameof(filterSize));
            if (stride < 1) throw new ArgumentException("Stride must be greater than 0", nameof(stride));
            FilterSize = filterSize;
            Stride = stride;
            UseBias = useBias;
            KernelInitializer = kernelInitializer ?? throw new ArgumentNullException(nameof(kernelInitializer));
            BiasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
        }

        public void Build(ISize inputSize)
        {
            if (FilterSize > inputSize.TotalSize) throw new ArgumentException("filterSize cannot be more than the size of the previous layer", nameof(inputSize));
            OutputSize = new Size1D(ConvolutionUtils.CalculateOutputSize(inputSize.TotalSize, FilterSize, Stride));

            Output = new float[OutputSize.TotalSize];
            Filter = KernelInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, FilterSize);
            if (UseBias)
            {
                Biases = BiasInitializer.GenerateValues(OutputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize);
            }
        }

        public float[] Update(float[] inputValues)
        {
            for (int kernelIndex = 0; kernelIndex < OutputSize.TotalSize; kernelIndex++)
            {
                var value = DotProductWithFilter(inputValues, kernelIndex * Stride);
                if (UseBias)
                {
                    value += Biases[kernelIndex];
                }
                Output[kernelIndex] = value;
            }

            return Output;
        }

        private float DotProductWithFilter(float[] inputValue, int windowStart)
        {
            var sum = 0f;
            for (int i = 0; i < FilterSize; i++)
            {
                sum += inputValue[windowStart + i] * Filter[i];
            }

            return sum;
        }
    }
}
