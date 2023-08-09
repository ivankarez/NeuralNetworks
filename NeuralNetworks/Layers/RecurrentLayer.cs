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

        private float[] nodeInputs;
        private ValueStoreRange[] weights;
        private ValueStoreRange recurrentWeights;
        private ValueStoreRange kernels;

        public RecurrentLayer(int nodeCount, IActivation activation)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            if (activation == null) throw new ArgumentNullException(nameof(activation));

            NodeCount = nodeCount;
            this.activation = activation;
        }

        public void Build(int prevLayerSize, ValueStore parameters, ValueStore state)
        {
            weights = new ValueStoreRange[NodeCount];
            for (int i = 0; i < NodeCount; i++)
            {
                weights[i] = parameters.AllocateRange(prevLayerSize);
            }
            recurrentWeights = parameters.AllocateRange(NodeCount);
            nodeInputs = new float[prevLayerSize + 1];
            kernels = state.AllocateRange(NodeCount);
        }

        public IValueArray Update(IValueArray inputValues)
        {
            var recurrentInputIndex = nodeInputs.Length - 1;
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
            {
                var nodeWeights = weights[nodeIndex];
                MathUtils.ElementwiseMultiply(inputValues, nodeWeights, nodeInputs);
                nodeInputs[recurrentInputIndex] = kernels[nodeIndex] * recurrentWeights[nodeIndex];
                kernels[nodeIndex] = activation.Apply(nodeInputs);
            }
            return kernels;
        }
    }
}
