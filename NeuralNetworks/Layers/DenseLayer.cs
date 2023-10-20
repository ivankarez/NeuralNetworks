using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DenseLayer : IModelLayer
    {
        public int NodeCount { get; }
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
            if (activation == null) throw new ArgumentNullException(nameof(activation));

            NodeCount = nodeCount;
            this.activation = activation;
            this.useBias = useBias;
            KernelInitializer = kernelInitializer;
            BiasInitializer = biasInitializer;
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(int inputSize)
        {
            weights = KernelInitializer.GenerateValues2d(inputSize, NodeCount, NodeCount, inputSize);
            nodeValues = new float[NodeCount];
            biases = useBias ? BiasInitializer.GenerateValues(inputSize, NodeCount, NodeCount) : new float[0];

            State.Add("nodeValues", nodeValues);
            Parameters.Add("biases", biases);
            Parameters.Add("weights", weights);
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
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
