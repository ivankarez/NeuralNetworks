using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class SigmoidActivation : IActivation
    {   
        public float Apply(float input)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-input));
        }
    }
}
