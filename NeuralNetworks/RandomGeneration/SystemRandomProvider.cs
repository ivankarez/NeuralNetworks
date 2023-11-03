using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.RandomGeneration
{
    public class SystemRandomProvider : IRandomProvider
    {
        public Random Random { get; }

        public SystemRandomProvider(Random random)
        {
            Random = random ?? throw new ArgumentNullException(nameof(random));
        }

        public float NextFloat()
        {
            return (float)Random.NextDouble();
        }
    }
}
