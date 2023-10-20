using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.RandomGeneration.Initializers
{
    public class ConstantInitializer : IInitializer
    {
        public float Value { get; }

        public ConstantInitializer(float value)
        {
            Value = value;
        }

        public float GenerateValue(int fanIn, int fanOut)
        {
            return Value;
        }
    }
}
