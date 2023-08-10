using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Extensions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks
{
    public class LayeredNetworkModel
    {
        private readonly IModelLayer[] layers;
        private readonly IValueArray outputArray;
        private readonly IValueArray inputArray;

        public int Inputs { get; }
        public ValueStore Parameters { get; }
        public ValueStore State { get; }

        public LayeredNetworkModel(int inputs, params IModelLayer[] layers)
        {
            if (inputs <= 0) throw new ArgumentOutOfRangeException(nameof(inputs), "Must be bigger than 0");
            if (layers.Length == 0) throw new ArgumentException("Must have at least 1 element", nameof(layers));

            Inputs = inputs;
            this.layers = layers;

            Parameters = new ValueStore();
            State = new ValueStore();

            Build();

            var outputSize = layers[^1].NodeCount;
            outputArray = new ValueArray(new float[outputSize]);
            inputArray = new ValueArray(new float[inputs]);
        }

        private void Build()
        {
            var inputSize = Inputs;
            foreach (var layer in layers)
            {
                layer.Build(inputSize, Parameters, State);
                inputSize = layer.NodeCount;
            }
        }

        public IReadonlyValueArray Feedforward(float[] inputValues)
        {
            if (inputValues.Length != Inputs) throw new ArgumentException($"Must have length of {Inputs}", nameof(inputValues));

            inputArray.SetValues(inputValues);
            var layerInputs = inputArray;
            foreach (var layer in layers)
            {
                layerInputs = layer.Update(layerInputs);
            }
            outputArray.SetValues(layerInputs);

            return outputArray;
        }
    }
}
