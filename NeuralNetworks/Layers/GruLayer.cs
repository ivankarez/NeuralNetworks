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

        public NamedVectors<float> Parameters { get; } = new NamedVectors<float>();
        public NamedVectors<float> State { get; } = new NamedVectors<float>();

        public float[,] ForgetGateWeights { get; private set; }
        public float[,] CandidateWeights { get; private set; }
        public float[] NodeValues { get; private set; }
        public float[] ForgetRecurrentWeights { get; private set; }
        public float[] CandidateRecurrentWeights { get; private set; }
        public float[] ForgetBiases { get; private set; }
        public float[] CandidateBiases { get; private set; }

        public GruLayer(Size1D nodeCount,
            IActivation activation,
            IActivation recurrentActivation,
            bool useBias,
            IInitializer kernelInitializer,
            IInitializer recurrentInitializer,
            IInitializer biasInitializer)
        {
            if (nodeCount == null) throw new ArgumentNullException(nameof(nodeCount));
            if (nodeCount.TotalSize <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount.TotalSize), "Must be bigger than 0");

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
            if (inputSize.TotalSize <= 0) throw new ArgumentOutOfRangeException(nameof(inputSize.TotalSize), "Must be bigger than 0");

            var inputs = inputSize.TotalSize;
            var nodes = OutputSize.TotalSize;

            ForgetGateWeights = KernelInitializer.GenerateValues2d(inputs, nodes, nodes, inputs);
            ForgetRecurrentWeights = RecurrentInitializer.GenerateValues(inputs, nodes, nodes);
            ForgetBiases = BiasInitializer.GenerateValues(inputs, nodes, nodes);

            CandidateWeights = KernelInitializer.GenerateValues2d(inputs, nodes, nodes, inputs);
            CandidateRecurrentWeights = RecurrentInitializer.GenerateValues(inputs, nodes, nodes);
            CandidateBiases = BiasInitializer.GenerateValues(inputs, nodes, nodes);

            NodeValues = new float[nodes];

            Parameters.Add("forgetGateWeights", ForgetGateWeights);
            Parameters.Add("forgetRecurrentWeights", ForgetRecurrentWeights);
            Parameters.Add("forgetBiases", ForgetBiases);
            Parameters.Add("candidateWeights", CandidateWeights);
            Parameters.Add("candidateRecurrentWeights", CandidateRecurrentWeights);
            Parameters.Add("candidateBiases", CandidateBiases);
            State.Add("nodeValues", NodeValues);
        }

        public float[] Update(float[] inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < OutputSize.TotalSize; nodeIndex++)
            {
                UpdateCell(nodeIndex, inputValues);
            }

            return NodeValues;
        }

        private void UpdateCell(int index, float[] inputs)
        {
            var forgetGateInput = Mutliply(ForgetGateWeights, inputs, index) + (NodeValues[index] * ForgetRecurrentWeights[index]) + ForgetBiases[index];
            var forgetGate = RecurrentActivation.Apply(forgetGateInput);
            var candidateInput = Mutliply(CandidateWeights, inputs, index) + (NodeValues[index] * CandidateRecurrentWeights[index] * forgetGate) + CandidateBiases[index];
            var candidate = Activation.Apply(candidateInput);
            NodeValues[index] = (1 - forgetGate) * NodeValues[index] + forgetGate * candidate;
        }

        private float Mutliply(float[,] weights, float[] inputs, int weightsIndex)
        {
            var result = 0f;
            for (int inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
            {
                result += weights[weightsIndex, inputIndex] * inputs[inputIndex];
            }

            return result;
        }
    }
}
