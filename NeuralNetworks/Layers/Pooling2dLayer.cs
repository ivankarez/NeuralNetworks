using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class Pooling2dLayer : IModelLayer
    {
        public int NodeCount { get; private set; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }
        public int InputWidth { get; }
        public int InputHeight { get; }
        public int WindowWidth { get; }
        public int WindowHeigth { get; }
        public int StrideX { get; }
        public int StrideY { get; }
        public PoolingType PoolingType { get; }

        private readonly Func<int, int, float[], float> pooling;
        private float[] nodeValues;
        private int nodeValuesWidth;
        private int nodeValuesHeight;

        public Pooling2dLayer(int inputWidth, int inputHeight, int windowWidth, int windowHeigth, int strideX, int strideY, PoolingType poolingType)
        {
            if (inputWidth < 1) throw new ArgumentException("Input width must be greater than 0", nameof(inputWidth));
            if (inputHeight < 1) throw new ArgumentException("Input height must be greater than 0", nameof(inputHeight));
            if (windowWidth < 2) throw new ArgumentException("Filter width must be greater than 1", nameof(windowWidth));
            if (windowHeigth < 2) throw new ArgumentException("Filter height must be greater than 1", nameof(windowHeigth));
            if (strideX < 1) throw new ArgumentException("Stride X must be greater than 0", nameof(strideX));
            if (strideY < 1) throw new ArgumentException("Stride Y must be greater than 0", nameof(strideY));

            if (windowWidth > inputWidth) throw new ArgumentException("Filter width cannot be greater than input width", nameof(windowWidth));
            if (windowHeigth > inputHeight) throw new ArgumentException("Filter height cannot be greater than input height", nameof(windowHeigth));

            InputWidth = inputWidth;
            InputHeight = inputHeight;
            WindowWidth = windowWidth;
            WindowHeigth = windowHeigth;
            StrideX = strideX;
            StrideY = strideY;
            PoolingType = poolingType;
            pooling = GetPooling();

            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(int inputSize)
        {
            var expectedInputSize = InputWidth * InputHeight;
            if (inputSize != expectedInputSize) throw new ArgumentException($"Input size must be {expectedInputSize}", nameof(inputSize));

            nodeValuesWidth = (InputWidth - WindowWidth) / StrideX + 1;
            nodeValuesHeight = (InputHeight - WindowHeigth) / StrideY + 1;
            NodeCount = nodeValuesWidth * nodeValuesHeight;
            nodeValues = new float[NodeCount];

            State.Add("nodeValues", nodeValues);
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeX = 0; nodeX < nodeValuesWidth; nodeX += 1)
            {
                for (int nodeY = 0; nodeY < nodeValuesHeight; nodeY += StrideY)
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
            for (int fx = 0; fx < WindowWidth; fx += 1)
            {
                for (int fy = 0; fy < WindowHeigth; fy += 1)
                {
                    var inputX = nodeX * StrideX + fx;
                    var inputY = nodeY * StrideY + fy;
                    var inputValue = inputValues[inputX * InputWidth + inputY];
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
            for (int fx = 0; fx < WindowWidth; fx += 1)
            {
                for (int fy = 0; fy < WindowHeigth; fy += 1)
                {
                    var inputX = nodeX * StrideX + fx;
                    var inputY = nodeY * StrideY + fy;
                    var inputValue = inputValues[inputX * InputWidth + inputY];
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
            for (int fx = 0; fx < WindowWidth; fx += 1)
            {
                for (int fy = 0; fy < WindowHeigth; fy += 1)
                {
                    var inputX = nodeX * StrideX + fx;
                    var inputY = nodeY * StrideY + fy;
                    var inputValue = inputValues[inputX * InputWidth + inputY];
                    nodeValue += inputValue;
                }
            }

            return nodeValue;
        }

        private float PoolByAverage(int nodeX, int nodeY, float[] inputValues)
        {
            return PoolBySum(nodeX, nodeY, inputValues) / (WindowWidth * WindowHeigth);
        }
    }
}
