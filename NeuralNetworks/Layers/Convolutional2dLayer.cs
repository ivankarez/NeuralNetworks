using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class Convolutional2dLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }
        public Size2D InputSize { get; set; }
        public Size2D FilterSize { get; }
        public Stride2D Stride { get; }
        public bool UseBias { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer BiasInitializer { get; }

        private float[] nodeValues;
        private float[] biases;
        private float[,] filter;
        private int outputWidth;
        private int outputHeight;

        public Convolutional2dLayer(Size2D filterSize, Stride2D stride,
            bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
        {
            if (filterSize == null) throw new ArgumentNullException(nameof(filterSize));
            if (stride == null) throw new ArgumentNullException(nameof(stride));
            if (kernelInitializer == null) throw new ArgumentNullException(nameof(kernelInitializer));
            if (biasInitializer == null) throw new ArgumentNullException(nameof(biasInitializer));

            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
            FilterSize = filterSize;
            Stride = stride;
            UseBias = useBias;
            KernelInitializer = kernelInitializer;
            BiasInitializer = biasInitializer;
        }

        public void Build(ISize inputSize)
        {
            if (!(inputSize is Size2D))
            {
                throw new ArgumentException($"Input size must be {nameof(Size2D)}", nameof(inputSize));
            }
            InputSize = inputSize as Size2D;
            outputWidth = ConvolutionUtils.CalculateOutputSize(InputSize.Width, FilterSize.Width, Stride.Horizontal);
            outputHeight = ConvolutionUtils.CalculateOutputSize(InputSize.Height, FilterSize.Height, Stride.Vertical);
            OutputSize = new Size2D(outputWidth, outputHeight);
            nodeValues = new float[OutputSize.TotalSize];
            filter = KernelInitializer.GenerateValues2d(inputSize.TotalSize, OutputSize.TotalSize, FilterSize.Width, FilterSize.Height);
            biases = UseBias ? BiasInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize) : new float[0];

            State.Add("nodeValues", nodeValues);
            Parameters.Add("filter", filter);
            Parameters.Add("biases", biases);
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeX = 0; nodeX < outputWidth; nodeX += 1)
            {
                for (int nodeY = 0; nodeY < outputHeight; nodeY += 1)
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
                    var nodeIndex = nodeX * outputHeight + nodeY;
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
