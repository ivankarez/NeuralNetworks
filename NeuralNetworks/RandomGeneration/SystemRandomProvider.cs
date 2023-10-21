using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.RandomGeneration
{
    public class SystemRandomProvider : IRandomProvider
    {
        public Random Random { get; }

        public SystemRandomProvider(Random random)
        {
            if (random == null) throw new ArgumentNullException(nameof(random));

            Random = random;
        }

        public float NextFloat()
        {
            return (float)Random.NextDouble();
        }
    }
}
