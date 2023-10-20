using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class Convolutional2dLayer : IModelLayer
    {
        public int NodeCount { get; private set; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }
        public int InputWidth { get; }
        public int InputHeight { get; }
        public int FilterWidth { get; }
        public int FilterHeigth { get; }
        public int StrideX { get; }
        public int StrideY { get; }

        private float[] nodeValues;
        private float[,] filter;
        private int nodeValuesWidth;
        private int nodeValuesHeight;

        public Convolutional2dLayer(int inputWidth, int inputHeight, int filterWidth, int filterHeigth, int strideX, int strideY)
        {
            if (inputWidth < 1) throw new ArgumentException("Input width must be greater than 0", nameof(inputWidth));
            if (inputHeight < 1) throw new ArgumentException("Input height must be greater than 0", nameof(inputHeight));
            if (filterWidth < 2) throw new ArgumentException("Filter width must be greater than 1", nameof(filterWidth));
            if (filterHeigth < 2) throw new ArgumentException("Filter height must be greater than 1", nameof(filterHeigth));
            if (strideX < 1) throw new ArgumentException("Stride X must be greater than 0", nameof(strideX));
            if (strideY < 1) throw new ArgumentException("Stride Y must be greater than 0", nameof(strideY));

            if (filterWidth > inputWidth) throw new ArgumentException("Filter width cannot be greater than input width", nameof(filterWidth));
            if (filterHeigth > inputHeight) throw new ArgumentException("Filter height cannot be greater than input height", nameof(filterHeigth));

            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
            InputWidth = inputWidth;
            InputHeight = inputHeight;
            FilterWidth = filterWidth;
            FilterHeigth = filterHeigth;
            StrideX = strideX;
            StrideY = strideY;
        }

        public void Build(int inputSize)
        {
            var expectedInputSize = InputWidth * InputHeight;
            if (inputSize != expectedInputSize) throw new ArgumentException($"Input size must be {expectedInputSize}", nameof(inputSize));

            nodeValuesWidth = (InputWidth - FilterWidth) / StrideX + 1;
            nodeValuesHeight = (InputHeight - FilterHeigth) / StrideY + 1;
            NodeCount = nodeValuesWidth * nodeValuesHeight;
            nodeValues = new float[NodeCount];
            filter = new float[FilterWidth, FilterHeigth];

            State.Add("nodeValues", nodeValues);
            Parameters.Add("filter", filter);
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeX = 0; nodeX < nodeValuesWidth; nodeX += 1)
            {
                for (int nodeY = 0; nodeY < nodeValuesHeight; nodeY += StrideY)
                {
                    var nodeValue = 0f;
                    for (int fx = 0; fx < filter.GetLength(0); fx += 1)
                    {
                        for (int fy = 0; fy < filter.GetLength(1); fy += 1)
                        {
                            var inputX = nodeX * StrideX + fx;
                            var inputY = nodeY * StrideY + fy;
                            nodeValue += inputValues[inputX * InputWidth + inputY] * filter[fx, fy];
                        }
                    }
                    var nodeIndex = nodeX * nodeValuesHeight + nodeY;
                    nodeValues[nodeIndex] = nodeValue;
                }
            }

            return nodeValues;
        }
    }
}
