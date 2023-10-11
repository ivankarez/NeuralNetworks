using Ivankarez.NeuralNetworks.Abstractions;
using System;
using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks
{
    public class LayeredNetworkModel
    {
        private readonly IModelLayer[] layers;

        public IReadOnlyList<IModelLayer> Layers => layers;
        public int Inputs { get; }

        public LayeredNetworkModel(int inputs, params IModelLayer[] layers)
        {
            if (inputs <= 0) throw new ArgumentOutOfRangeException(nameof(inputs), "Must be bigger than 0");
            if (layers.Length == 0) throw new ArgumentException("Must have at least 1 element", nameof(layers));

            Inputs = inputs;
            this.layers = layers;

            Build();
        }

        private void Build()
        {
            var inputSize = Inputs;
            foreach (var layer in layers)
            {
                layer.Build(inputSize);
                inputSize = layer.NodeCount;
            }
        }

        public float[] Feedforward(float[] inputValues)
        {
            if (inputValues.Length != Inputs) throw new ArgumentException($"Must have length of {Inputs}", nameof(inputValues));

            var layerInputs = inputValues;
            foreach (var layer in layers)
            {
                layerInputs = layer.Update(layerInputs);
            }

            return layerInputs;
        }
    }
}
