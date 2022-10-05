using NeuralNetworks.Abstractions;
using System;

namespace NeuralNetworks
{
    public class LayeredNetworkModel
    {
        private readonly IModelLayer[] layers;
        private readonly ModelParameters parameters;
        private readonly ModelParameters values;
        private readonly float[] outputArray;

        public int Inputs { get; }
        public ModelParameters Parameters => parameters;

        public LayeredNetworkModel(int inputs, params IModelLayer[] layers)
        {
            if (inputs <= 0) throw new ArgumentOutOfRangeException(nameof(inputs), "Must be bigger than 0");
            Inputs = inputs;

            if (layers.Length == 0) throw new ArgumentException("Must have at least 1 element", nameof(layers));
            this.layers = layers;

            parameters = new ModelParameters();
            values = new ModelParameters();
            outputArray = new float[layers[layers.Length - 1].NodeCount];

            Build();
        }

        private void Build()
        {
            var prevLayerSize = Inputs;
            foreach (var layer in layers)
            {
                layer.Build(prevLayerSize, parameters, values);
                prevLayerSize = layer.NodeCount;
            }
        }

        public float[] Predict(float[] inputValues)
        {
            if (inputValues.Length != Inputs) throw new ArgumentException($"Must have length of {Inputs}", nameof(inputValues));

            var prevLayerOutputs = new ParameterRange(0, inputValues.Length, inputValues);
            foreach (var layer in layers)
            {
                prevLayerOutputs = layer.Update(prevLayerOutputs);
            }
            prevLayerOutputs.CopyTo(outputArray);

            return outputArray; // Todo use custom read-only return type
        }
    }
}
