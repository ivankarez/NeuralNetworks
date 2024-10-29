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

        public IActivation Activation { get; set; }
        public bool UseBias { get; set; }

        public float[,] Weights { get; set; }
        public float[] RecurrentWeights { get; set; }
        public float[] Output { get; set; }
        public float[] Biases { get; set; }

        public RecurrentLayer(int nodeCount, IActivation activation, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer, IInitializer recurrentInitializer)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            OutputSize = new Size1D(nodeCount);
            Activation = activation ?? throw new ArgumentNullException(nameof(activation));
            UseBias = useBias;
            KernelInitializer = kernelInitializer ?? throw new ArgumentNullException(nameof(kernelInitializer));
            BiasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
            RecurrentInitializer = recurrentInitializer ?? throw new ArgumentNullException(nameof(recurrentInitializer));
        }

        public void Build(ISize inputSize)
        {
            Weights = KernelInitializer.GenerateValueMatrix(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize, inputSize.TotalSize);
            RecurrentWeights = RecurrentInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize);
            Output = new float[OutputSize.TotalSize];
            if (UseBias)
            {
                Biases = BiasInitializer.GenerateValues(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize);
            }
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < OutputSize.TotalSize; nodeIndex++)
            {
                UpdateNode(nodeIndex, inputValues);
            }
            return Output;
        }

        private void UpdateNode(int nodeIndex, float[] inputValues)
        {
            var nodeValue = RecurrentWeights[nodeIndex] * Output[nodeIndex];
            for (int inputIndex = 0; inputIndex < inputValues.Length; inputIndex++)
            {
                nodeValue += inputValues[inputIndex] * Weights[nodeIndex, inputIndex];
            }
            if (UseBias)
            {
                nodeValue += Biases[nodeIndex];
            }
            Output[nodeIndex] = Activation.Apply(nodeValue);
        }
    }
}
