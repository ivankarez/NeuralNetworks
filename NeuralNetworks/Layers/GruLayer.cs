using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class GruLayer : IModelLayer
    {
        public ISize OutputSize { get; }
        public IActivation Activation { get; }
        public IActivation RecurrentActivation { get; }
        public bool UseBias { get; }
        public IInitializer KernelInitializer { get; }
        public IInitializer RecurrentInitializer { get; }
        public IInitializer BiasInitializer { get; }

        public NamedVectors<float> Parameters { get; } = null;
        public NamedVectors<float> State { get; } = null;

        public float[][] ForgetGateWeights { get; set; }
        public float[][] CandidateWeights { get; set; }
        public float[] ForgetRecurrentWeights { get; set; }
        public float[] CandidateRecurrentWeights { get; set; }
        public float[] ForgetBiases { get; set; }
        public float[] CandidateBiases { get; set; }
        public float[] Output { get; set; }

        public GruLayer(Size1D nodeCount,
            IActivation activation,
            IActivation recurrentActivation,
            bool useBias,
            IInitializer kernelInitializer,
            IInitializer recurrentInitializer,
            IInitializer biasInitializer)
        {
            if (nodeCount == null) throw new ArgumentNullException(nameof(nodeCount));

            OutputSize = nodeCount ?? throw new ArgumentNullException(nameof(nodeCount));
            Activation = activation ?? throw new ArgumentNullException(nameof(activation));
            RecurrentActivation = recurrentActivation ?? throw new ArgumentNullException(nameof(recurrentActivation));
            UseBias = useBias;
            KernelInitializer = kernelInitializer ?? throw new ArgumentNullException(nameof(kernelInitializer));
            RecurrentInitializer = recurrentInitializer ?? throw new ArgumentNullException(nameof(recurrentInitializer));
            BiasInitializer = biasInitializer ?? throw new ArgumentNullException(nameof(biasInitializer));
        }

        public void Build(ISize inputSize)
        {
            if (inputSize == null) throw new ArgumentNullException(nameof(inputSize));

            var inputs = inputSize.TotalSize;
            var nodes = OutputSize.TotalSize;

            ForgetGateWeights = KernelInitializer.GenerateValue2D(inputs, nodes, nodes, inputs);
            ForgetRecurrentWeights = RecurrentInitializer.GenerateValues(inputs, nodes, nodes);

            CandidateWeights = KernelInitializer.GenerateValue2D(inputs, nodes, nodes, inputs);
            CandidateRecurrentWeights = RecurrentInitializer.GenerateValues(inputs, nodes, nodes);

            Output = new float[nodes];

            if (UseBias)
            {
                CandidateBiases = BiasInitializer.GenerateValues(inputs, nodes, nodes);
                ForgetBiases = BiasInitializer.GenerateValues(inputs, nodes, nodes);
            }
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < OutputSize.TotalSize; nodeIndex++)
            {
                UpdateCell(nodeIndex, inputValues);
            }

            return Output;
        }

        private void UpdateCell(int index, float[] inputs)
        {
            var forgetGateInput = Mutliply(ForgetGateWeights[index], inputs) + (Output[index] * ForgetRecurrentWeights[index]);
            if (UseBias)
            {
                forgetGateInput += ForgetBiases[index];
            }
            var forgetGate = RecurrentActivation.Apply(forgetGateInput);

            var candidateInput = Mutliply(CandidateWeights[index], inputs) + (Output[index] * CandidateRecurrentWeights[index] * forgetGate);
            if (UseBias)
            {
                candidateInput += CandidateBiases[index];
            }
            var candidate = Activation.Apply(candidateInput);
            Output[index] = (1 - forgetGate) * Output[index] + forgetGate * candidate;
        }

        private float Mutliply(float[] weights, float[] inputs)
        {
            var result = 0f;
            for (int inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
            {
                result += weights[inputIndex] * inputs[inputIndex];
            }

            return result;
        }
    }
}
