using NeuralNetworks.Abstractions;
using System;

namespace NeuralNetworks.Layers
{
    public class DenseLayer : IModelLayer
    {
        public int NodeCount { get; }

        private readonly IActivation activation;
        private readonly float[] values;

        private ParameterRange[] weights;

        public DenseLayer(int nodeCount, IActivation activation)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            if (activation == null) throw new ArgumentNullException(nameof(activation));

            NodeCount = nodeCount;
            values = new float[NodeCount];
            this.activation = activation;
        }

        public void Build(int prevLayerSize, ModelParameters parameterBuilder, ModelParameters stateBuilder)
        {
            weights = new ParameterRange[NodeCount];
            for (int i = 0; i < NodeCount; i++)
            {
                weights[i] = parameterBuilder.AllocateRange(prevLayerSize);
            }
        }

        public float[] Update(float[] inputValues)
        {
            var nodeInputs = new float[inputValues.Length];
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
            {
                var nodeWeights = weights[nodeIndex];
                for (int inputIndex = 0; inputIndex < inputValues.Length; inputIndex++)
                {
                    nodeInputs[inputIndex] = inputValues[inputIndex] * nodeWeights[inputIndex];
                }
                values[nodeIndex] = activation.Apply(nodeInputs);
            }
            return values;
        }
    }
}
