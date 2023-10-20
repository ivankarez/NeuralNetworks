using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.RandomGeneration.Initializers
{
    public class NormalInitializer : IInitializer
    {
        public IRandomProvider RandomProvider { get; }
        public float Mean { get; }
        public float StdDev { get; }

        public NormalInitializer(IRandomProvider randomProvider, float mean, float stdDev)
        {
            RandomProvider = randomProvider;
            Mean = mean;
            StdDev = stdDev;
        }

        public float GenerateValue(int fanIn, int fanOut)
        {
            return RandomProvider.NextNormalFloat(Mean, StdDev);
        }
    }
}
