using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DenseLayer : IModelLayer
    {
        public ISize OutputSize { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer BiasInitializer { get; }

        public IActivation Activation { get; set; }
        public float[] Output { get; private set; }
        public float[][] Weights { get; private set; }
        public float[] Biases { get; private set; }
        public bool UseBias;


        public DenseLayer(int nodeCount, IActivation activation, bool useBias, IInitializer kernelInitializer, IInitializer biasInitializer)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");

            OutputSize = new Size1D(nodeCount);
            Activation = activation ?? throw new ArgumentNullException(nameof(activation));
            UseBias = useBias;
            KernelInitializer = kernelInitializer;
            BiasInitializer = biasInitializer;
        }

        public void Build(ISize inputSize)
        {
            Weights = KernelInitializer.GenerateValue2D(inputSize.TotalSize, OutputSize.TotalSize, OutputSize.TotalSize, inputSize.TotalSize);
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
            var nodeValue = 0f;
            for (int inputIndex = 0; inputIndex < inputValues.Length; inputIndex++)
            {
                nodeValue += inputValues[inputIndex] * Weights[nodeIndex][inputIndex];
            }
            if (UseBias)
            {
                nodeValue += Biases[nodeIndex];
            }
            Output[nodeIndex] = Activation.Apply(nodeValue);
        }
    }
}
