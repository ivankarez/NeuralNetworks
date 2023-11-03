using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class Pooling2dLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; }
        public Size2D InputSize { get; set; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }
        public Size2D WindowSize { get; }
        public Stride2D Stride { get; }
        public PoolingType PoolingType { get; }

        private readonly Func<int, int, float[], float> pooling;
        private float[] nodeValues;
        private int nodeValuesWidth;
        private int nodeValuesHeight;

        public Pooling2dLayer(Size2D windowSize, Stride2D stride, PoolingType poolingType)
        {
            WindowSize = windowSize ?? throw new ArgumentNullException(nameof(windowSize));
            Stride = stride ?? throw new ArgumentNullException(nameof(stride));
            PoolingType = poolingType;
            pooling = GetPooling();

            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(ISize inputSize)
        {
            if (!(inputSize is Size2D)) throw new ArgumentException($"Input size must be {nameof(Size2D)}", nameof(inputSize));
            InputSize = inputSize as Size2D;

            nodeValuesWidth = ConvolutionUtils.CalculateOutputSize(InputSize.Width, WindowSize.Width, Stride.Horizontal);
            nodeValuesHeight = ConvolutionUtils.CalculateOutputSize(InputSize.Height, WindowSize.Height, Stride.Vertical);
            OutputSize = new Size2D(nodeValuesWidth, nodeValuesHeight);
            nodeValues = new float[OutputSize.TotalSize];

            State.Add("nodeValues", nodeValues);
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeX = 0; nodeX < nodeValuesWidth; nodeX += 1)
            {
                for (int nodeY = 0; nodeY < nodeValuesHeight; nodeY += 1)
                {
                    var nodeValue = pooling(nodeX, nodeY, inputValues);
                    var nodeIndex = nodeX * nodeValuesHeight + nodeY;
                    nodeValues[nodeIndex] = nodeValue;
                }
            }

            return nodeValues;
        }

        private Func<int, int, float[], float> GetPooling()
        {
            return PoolingType switch
            {
                PoolingType.Max => PoolByMax,
                PoolingType.Average => PoolByAverage,
                PoolingType.Min => PoolByMin,
                PoolingType.Sum => PoolBySum,
                _ => throw new NotImplementedException($"Unknown pooling {PoolingType}"),
            };
        }

        private float PoolByMax(int nodeX, int nodeY, float[] inputValues)
        {
            var nodeValue = float.NegativeInfinity;
            for (int fx = 0; fx < WindowSize.Width; fx += 1)
            {
                for (int fy = 0; fy < WindowSize.Height; fy += 1)
                {
                    var inputX = nodeX * Stride.Horizontal + fx;
                    var inputY = nodeY * Stride.Vertical + fy;
                    var inputValue = inputValues[inputX * InputSize.Width + inputY];
                    if (inputValue > nodeValue)
                    {
                        nodeValue = inputValue;
                    }
                }
            }

            return nodeValue;
        }

        private float PoolByMin(int nodeX, int nodeY, float[] inputValues)
        {
            var nodeValue = float.PositiveInfinity;
            for (int fx = 0; fx < WindowSize.Width; fx += 1)
            {
                for (int fy = 0; fy < WindowSize.Height; fy += 1)
                {
                    var inputX = nodeX * Stride.Horizontal + fx;
                    var inputY = nodeY * Stride.Vertical + fy;
                    var inputValue = inputValues[inputX * InputSize.Width + inputY];
                    if (inputValue < nodeValue)
                    {
                        nodeValue = inputValue;
                    }
                }
            }

            return nodeValue;
        }

        private float PoolBySum(int nodeX, int nodeY, float[] inputValues)
        {
            var nodeValue = 0f;
            for (int fx = 0; fx < WindowSize.Width; fx += 1)
            {
                for (int fy = 0; fy < WindowSize.Height; fy += 1)
                {
                    var inputX = nodeX * Stride.Horizontal + fx;
                    var inputY = nodeY * Stride.Vertical + fy;
                    var inputValue = inputValues[inputX * InputSize.Width + inputY];
                    nodeValue += inputValue;
                }
            }

            return nodeValue;
        }

        private float PoolByAverage(int nodeX, int nodeY, float[] inputValues)
        {
            return PoolBySum(nodeX, nodeY, inputValues) / (WindowSize.Width * WindowSize.Height);
        }
    }
}
