using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.RandomGeneration.Initializers
{
    internal class UniformInitializer : IInitializer
    {
        public IRandomProvider RandomProvider { get; }
        public float Min { get; }
        public float Max { get; }

        public UniformInitializer(IRandomProvider randomProvider, float min, float max)
        {
            RandomProvider = randomProvider;
            Min = min;
            Max = max;
        }

        public float GenerateValue(int fanIn, int fanOut)
        {
            return RandomProvider.NextFloat(Min, Max);
        }
    }
}
