using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class RecurrentLayer : IModelLayer
    {
        public ISize OutputSize { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer BiasInitializer { get; }
        public IInitializer RecurrentInitializer { get; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        private readonly IActivation activation;
        private readonly bool useBias;

        private float[,] weights;
        private float[] recurrentWeights;
        private float[] nodeValues;
        private float[] biases;

        public RecurrentLayer(int nodeCount, IActivation activation, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer, IInitializer recurrentInitializer)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            OutputSize = new Size1D(nodeCount);
            this.activation = activation ?? throw new ArgumentNullException(nameof(activation));
            this.useBias = useBias;
            KernelInitializer = kernelInitializer ?? throw new ArgumentNullException(nameof(kernelInitializer));
            BiasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
            RecurrentInitializer = recurrentInitializer ?? throw new ArgumentNullException(nameof(recurrentInitializer));
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(ISize inputSize)
        {
            weights = KernelInitializer.GenerateValues2d(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize, inputSize.TotalSize);
            recurrentWeights = RecurrentInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize);
            nodeValues = new float[OutputSize.TotalSize];
            if (useBias)
            {
                biases = BiasInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize);
                Parameters.Add("biases", biases);
            }

            State.Add("nodeValues", nodeValues);
            Parameters.Add("weights", weights);
            Parameters.Add("recurrentWeights", recurrentWeights);
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
            var nodeValue = recurrentWeights[nodeIndex] * nodeValues[nodeIndex];
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
