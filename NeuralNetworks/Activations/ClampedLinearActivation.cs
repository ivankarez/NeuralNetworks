using NeuralNetworks.Abstractions;
using System;

namespace NeuralNetworks.Activations
{
    public class ClampedLinearActivation : IActivation
    {
        public float Min { get; }
        public float Max { get; }

        private readonly LinearActivation internalActivation = new LinearActivation();

        public ClampedLinearActivation(float min, float max)
        {
            if (min >= max) throw new ArgumentException("Minimum value cannot be bigger than maximum value", nameof(min));

            Min = min;
            Max = max;
        }

        public float Apply(float[] inputs)
        {
            var value = internalActivation.Apply(inputs);

            if (value > Max) return Max;
            if (value < Min) return Min;
            return value;
        }
    }
}
