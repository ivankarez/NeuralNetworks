using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class SigmoidActivation : IActivation
    {
        public float Apply(float[] inputs)
        {
            var sum = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                sum += 1.0f / (1.0f + (float)Math.Exp(-inputs[i]));
            }

            return sum;
        }
    }
}
