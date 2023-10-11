using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class LinearActivation : IActivation
    {
        public float Apply(float input)
        {
            return input;
        }
    }
}
