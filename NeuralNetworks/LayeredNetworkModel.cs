using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Extensions;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks
{
    public class LayeredNetworkModel
    {
        private readonly IModelLayer[] layers;
        private readonly ValueStore parameters;
        private readonly ValueStore state;
        private readonly IValueArray outputArray;
        private readonly IValueArray inputArray;

        public int Inputs { get; }
        public ValueStore Parameters => parameters;
        public ValueStore State => state;

        public LayeredNetworkModel(int inputs, params IModelLayer[] layers)
        {
            if (inputs <= 0) throw new ArgumentOutOfRangeException(nameof(inputs), "Must be bigger than 0");
            Inputs = inputs;

            if (layers.Length == 0) throw new ArgumentException("Must have at least 1 element", nameof(layers));
            this.layers = layers;

            parameters = new ValueStore();
            state = new ValueStore();
            var outputSize = layers[layers.Length - 1].NodeCount;
            outputArray = new ValueArray(new float[outputSize]);
            inputArray = new ValueArray(new float[inputs]);

            Build();
        }

        private void Build()
        {
            var prevLayerSize = Inputs;
            foreach (var layer in layers)
            {
                layer.Build(prevLayerSize, parameters, state);
                prevLayerSize = layer.NodeCount;
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
