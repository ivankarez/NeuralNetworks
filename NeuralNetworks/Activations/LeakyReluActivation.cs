using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class LeakyReluActivation : IActivation
    {
        private readonly float alpha;

        public LeakyReluActivation(float alpha) {
            this.alpha = alpha;
        }

        public float Apply(float[] inputs)
        {
            var sum = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] > 0f)
                {
                    sum += inputs[i];
                }
                else
                {
                    sum += inputs[i] * alpha;
                }
            }
            return sum;
        }
    }
}
