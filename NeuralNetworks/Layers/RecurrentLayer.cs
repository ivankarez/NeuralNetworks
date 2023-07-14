using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class RecurrentLayer : IModelLayer
    {
        public int NodeCount { get; }

        private readonly IActivation activation;

        private float[] nodeInputs;
        private ParameterRange[] weights;
        private ParameterRange kernels;

        public RecurrentLayer(int nodeCount, IActivation activation)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            if (activation == null) throw new ArgumentNullException(nameof(activation));

            NodeCount = nodeCount;
            this.activation = activation;
        }

        public void Build(int prevLayerSize, ModelParameters parameterBuilder, ModelParameters valueBuilder)
        {
            weights = new ParameterRange[NodeCount];
            for (int i = 0; i < NodeCount; i++)
            {
                weights[i] = parameterBuilder.AllocateRange(prevLayerSize + 1);
            }
            nodeInputs = new float[prevLayerSize + 1];
            kernels = valueBuilder.AllocateRange(NodeCount);
        }

        public ParameterRange Update(ParameterRange inputValues)
        {
            var recurrentInputIndex = nodeInputs.Length - 1;
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
            {
                var nodeWeights = weights[nodeIndex];
                for (int inputIndex = 0; inputIndex < inputValues.Size; inputIndex++)
                {
                    nodeInputs[inputIndex] = inputValues[inputIndex] * nodeWeights[inputIndex];
                }
                nodeInputs[recurrentInputIndex] = kernels[nodeIndex] * nodeWeights[recurrentInputIndex];
                kernels[nodeIndex] = activation.Apply(nodeInputs);
            }
            return kernels;
        }
    }
}
