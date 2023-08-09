using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class RecurrentLayer : IModelLayer
    {
        public int NodeCount { get; }

        private readonly IActivation activation;
        private readonly bool useBias;

        private float[] nodeInputs;
        private ValueStoreRange[] weights;
        private ValueStoreRange recurrentWeights;
        private ValueStoreRange kernels;

        public RecurrentLayer(int nodeCount, IActivation activation, bool useBias)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            if (activation == null) throw new ArgumentNullException(nameof(activation));

            NodeCount = nodeCount;
            this.activation = activation;
            this.useBias = useBias;
        }

        public void Build(int inputSize, ValueStore parameters, ValueStore state)
        {
            var nodeInputSize = useBias ? inputSize + 1 : inputSize;
            weights = new ValueStoreRange[NodeCount];
            for (int i = 0; i < NodeCount; i++)
            {
                weights[i] = parameters.AllocateRange(nodeInputSize);
            }
            recurrentWeights = parameters.AllocateRange(NodeCount);
            nodeInputs = new float[nodeInputSize + 1];
            kernels = state.AllocateRange(NodeCount);
        }

        public IValueArray Update(IValueArray inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
            {
                var nodeWeights = weights[nodeIndex];
                MathUtils.ElementwiseMultiply(inputValues, nodeWeights, nodeInputs);
                if (useBias)
                {
                    SetBiasInput(nodeWeights[^1]);
                }
                SetRecurrentInput(kernels[nodeIndex] * recurrentWeights[nodeIndex]);
                kernels[nodeIndex] = activation.Apply(nodeInputs);
            }
            return kernels;
        }

        private void SetRecurrentInput(float value)
        {
            nodeInputs[^1] = value;
        }

        private void SetBiasInput(float value)
        {
            nodeInputs[^2] = value;
        }
    }
}
