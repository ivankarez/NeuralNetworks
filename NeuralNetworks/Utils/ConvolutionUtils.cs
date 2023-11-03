using System;

namespace Ivankarez.NeuralNetworks.Utils
{
    public static class ConvolutionUtils
    {
        public static int CalculateOutputSize(int inputSize, int window, int stride)
        {
            if (inputSize <= 0) throw new ArgumentException("Input size must be greater than 0", nameof(inputSize));
            if (window <= 0) throw new ArgumentException("Window size must be greater than 0", nameof(window));
            if (stride <= 0) throw new ArgumentException("Stride must be greater than 0", nameof(stride));
            if (inputSize < window) throw new ArgumentException("Window size must be less than input size", nameof(window));

            return (inputSize - window) / stride + 1;
        }
    }
}
