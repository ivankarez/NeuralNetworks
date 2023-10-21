using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class TanhActivation : IActivation
    {
        public float Apply(float input)
        {
            return (float)Math.Tanh(input);
        }
    }
}
