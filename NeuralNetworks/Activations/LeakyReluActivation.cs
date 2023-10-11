using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class LeakyReluActivation : IActivation
    {
        private readonly float alpha;

        public LeakyReluActivation(float alpha) {
            this.alpha = alpha;
        }

        public float Apply(float input)
        {
            if (input < 0)
            {
                return alpha * input;
            }

            return input;
        }
    }
}
