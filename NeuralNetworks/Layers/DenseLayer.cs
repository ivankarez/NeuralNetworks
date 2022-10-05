﻿using NeuralNetworks.Abstractions;
using System;

namespace NeuralNetworks.Layers
{
    public class DenseLayer : IModelLayer
    {
        public int NodeCount { get; }

        private readonly IActivation activation;

        private ParameterRange[] weights;
        private ParameterRange kernels;
        private float[] nodeInputs;

        public DenseLayer(int nodeCount, IActivation activation)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            if (activation == null) throw new ArgumentNullException(nameof(activation));

            NodeCount = nodeCount;
            this.activation = activation;
        }

        public void Build(int prevLayerSize, ModelParameters parameterBuilder, ModelParameters stateBuilder)
        {
            weights = new ParameterRange[NodeCount];
            for (int i = 0; i < NodeCount; i++)
            {
                weights[i] = parameterBuilder.AllocateRange(prevLayerSize);
            }
            kernels = stateBuilder.AllocateRange(NodeCount);
            nodeInputs = new float[prevLayerSize];
        }

        public ParameterRange Update(ParameterRange inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
            {
                var nodeWeights = weights[nodeIndex];
                for (int inputIndex = 0; inputIndex < inputValues.Size; inputIndex++)
                {
                    nodeInputs[inputIndex] = inputValues[inputIndex] * nodeWeights[inputIndex];
                }
                kernels[nodeIndex] = activation.Apply(nodeInputs);
            }
            return kernels;
        }
    }
}