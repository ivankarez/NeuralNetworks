using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Activations
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

        public float Apply(float input)
        {
            if (input > Max) return Max;
            if (input < Min) return Min;
            return input;
        }
    }
}
