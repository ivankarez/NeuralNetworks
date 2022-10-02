using NeuralNetworks.Abstractions;

namespace NeuralNetworks.Activations
{
    public class LinearActivation : IActivation
    {
        public float Apply(float[] inputs)
        {
            var sum = 0f;
            for (int i = 0; i < inputs.Length; i++)
            {
                sum += inputs[i];
            }

            return sum;
        }
    }
}
