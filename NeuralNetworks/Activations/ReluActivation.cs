using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Activations
{
    public class ReluActivation : IActivation
    {
        public float Apply(float[] inputs)
        {
            var sum = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] > 0f)
                {
                    sum += inputs[i];
                }
            }

            return sum;
        }
    }
}
