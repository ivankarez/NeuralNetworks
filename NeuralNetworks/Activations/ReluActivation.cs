using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class ReluActivation : IActivation
    {
        public float Apply(float input)
        {
            return input > 0f ? input : 0f;
        }
    }
}
