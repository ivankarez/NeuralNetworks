using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DenseLayer : IModelLayer
    {
        public int NodeCount { get; }

        private readonly IActivation activation;

        private ValueStoreRange[] weights;
        private ValueStoreRange kernels;
        private float[] nodeInputs;
        private readonly bool useBias;

        public DenseLayer(int nodeCount, IActivation activation, bool useBias)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            if (activation == null) throw new ArgumentNullException(nameof(activation));

            NodeCount = nodeCount;
            this.activation = activation;
            this.useBias = useBias;
        }

        public void Build(int prevLayerSize, ValueStore parameters, ValueStore state)
        {
            var nodeInputSize = useBias ? prevLayerSize + 1 : prevLayerSize;
            weights = new ValueStoreRange[NodeCount];
            for (int i = 0; i < NodeCount; i++)
            {
                weights[i] = parameters.AllocateRange(nodeInputSize);
            }
            kernels = state.AllocateRange(NodeCount);
            nodeInputs = new float[nodeInputSize];
        }

        public IValueArray Update(IValueArray inputValues)
        {
            var biasIndex = nodeInputs.Length - 1;
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
            {
                var nodeWeights = weights[nodeIndex];
                for (int inputIndex = 0; inputIndex < inputValues.Count; inputIndex++)
                {
                    nodeInputs[inputIndex] = inputValues[inputIndex] * nodeWeights[inputIndex];
                }
                if (useBias)
                {
                    nodeInputs[biasIndex] = nodeWeights[biasIndex];
                }
                kernels[nodeIndex] = activation.Apply(nodeInputs);
            }
            return kernels;
        }
    }
}
