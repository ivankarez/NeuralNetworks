using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class PoolingLayer : IModelLayer
    {
        public int NodeCount { get; private set; }
        public int Window { get; }
        public int Stride { get; }
        public PoolingType Type { get; }

        private ValueStoreRange kernels;

        public PoolingLayer(int window, int stride, PoolingType type)
        {
            if (window < 1) throw new ArgumentException("Window must be greater than 0", nameof(window));
            if (stride < 1) throw new ArgumentException("Stride must be greater than 0", nameof(stride));

            Window = window;
            Stride = stride;
            Type = type;
            NodeCount = -1;
        }

        public void Build(int inputSize, ValueStore parameters, ValueStore state)
        {
            NodeCount = (int)Math.Ceiling((double)inputSize / Stride);
            kernels = state.AllocateRange(NodeCount);
        }

        public IValueArray Update(IValueArray inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < kernels.Count; nodeIndex++)
            {
                var startIndex = nodeIndex * Stride;
                if (Type == PoolingType.Max)
                {
                    kernels[nodeIndex] = PoolByMaximum(startIndex, inputValues);
                }
                else if (Type == PoolingType.Average)
                {
                    kernels[nodeIndex] = PoolByAverage(startIndex, inputValues);
                }
                else if (Type == PoolingType.Min)
                {
                    kernels[nodeIndex] = PoolByMinimum(startIndex, inputValues);
                }
                else if (Type == PoolingType.Sum)
                {
                    kernels[nodeIndex] = PoolBySum(startIndex, inputValues);
                }
            }

            return kernels;
        }

        private float PoolByMaximum(int start, IValueArray inputValues)
        {
            var windowEnd = Math.Min(start + Window, inputValues.Count);
            var max = float.NegativeInfinity;
            for (int i = start; i < windowEnd; i++)
            {
                var value = inputValues[i];
                if (value > max)
                {
                    max = value;
                }
            }

            return max;
        }

        private float PoolByMinimum(int start, IValueArray inputValues)
        {
            var windowEnd = Math.Min(start + Window, inputValues.Count);
            var min = float.PositiveInfinity;
            for (int i = start; i < windowEnd; i++)
            {
                var value = inputValues[i];
                if (value < min)
                {
                    min = value;
                }
            }
            return min;
        }

        private float PoolByAverage(int start, IValueArray inputValues)
        {
            var windowEnd = Math.Min(start + Window, inputValues.Count);
            return PoolBySum(start, inputValues) / (windowEnd - start);
        }

        private float PoolBySum(int start, IValueArray inputValues)
        {
            var windowEnd = Math.Min(start + Window, inputValues.Count);
            var sum = 0f;
            for (int i = start; i < windowEnd; i++)
            {
                sum += inputValues[i];
            }
            return sum;
        }
    }

    public enum PoolingType
    {
        Max,
        Average,
        Min,
        Sum
    }
}
