using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DenseLayer : IModelLayer
    {
        public int NodeCount { get; }

        private readonly IActivation activation;

        private ParameterRange[] weights;
        private ParameterRange kernels;
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

        public void Build(int prevLayerSize, ModelParameters parameterBuilder, ModelParameters stateBuilder)
        {
            var nodeInputSize = useBias ? prevLayerSize + 1 : prevLayerSize;
            weights = new ParameterRange[NodeCount];
            for (int i = 0; i < NodeCount; i++)
            {
                weights[i] = parameterBuilder.AllocateRange(nodeInputSize);
            }
            kernels = stateBuilder.AllocateRange(NodeCount);
            nodeInputs = new float[nodeInputSize];
        }

        public ParameterRange Update(ParameterRange inputValues)
        {
            var biasIndex = nodeInputs.Length - 1;
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
            {
                var nodeWeights = weights[nodeIndex];
                for (int inputIndex = 0; inputIndex < inputValues.Size; inputIndex++)
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
