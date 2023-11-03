using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.RandomGeneration.Initializers
{
    public class GlorotUniformInitializer : IInitializer
    {
        public IRandomProvider RandomProvider { get; }

        public GlorotUniformInitializer(IRandomProvider randomProvider) 
        {
            RandomProvider = randomProvider ?? throw new ArgumentNullException(nameof(randomProvider));
        }

        public float GenerateValue(int fanIn, int fanOut)
        {
            var limit = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
            return RandomProvider.NextFloat(-limit, limit);
        }
    }
}
