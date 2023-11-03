using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DenseLayer : IModelLayer
    {
        public ISize OutputSize { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer BiasInitializer { get; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        private readonly IActivation activation;

        private float[,] weights;
        private float[] nodeValues;
        private float[] biases;
        private readonly bool useBias;

        public DenseLayer(int nodeCount, IActivation activation, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            OutputSize = new Size1D(nodeCount);
            this.activation = activation ?? throw new ArgumentNullException(nameof(activation));
            this.useBias = useBias;
            KernelInitializer = kernelInitializer;
            BiasInitializer = biasInitializer;
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(ISize inputSize)
        {
            weights = KernelInitializer.GenerateValues2d(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize, inputSize.TotalSize);
            nodeValues = new float[OutputSize.TotalSize];
            biases = useBias ? BiasInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize) : new float[0];

            State.Add("nodeValues", nodeValues);
            Parameters.Add("biases", biases);
            Parameters.Add("weights", weights);
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < OutputSize.TotalSize; nodeIndex++)
            {
                UpdateNode(nodeIndex, inputValues);
            }
            return nodeValues;
        }

        private void UpdateNode(int nodeIndex, float[] inputValues)
        {
            var nodeValue = 0f;
            for (int inputIndex = 0; inputIndex < inputValues.Length; inputIndex++)
            {
                nodeValue += inputValues[inputIndex] * weights[nodeIndex, inputIndex];
            }
            if (useBias)
            {
                nodeValue += biases[nodeIndex];
            }
            nodeValues[nodeIndex] = activation.Apply(nodeValue);
        }
    }
}
