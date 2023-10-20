using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.RandomGeneration
{
    public class SystemRandomProvider : IRandomProvider
    {
        private readonly Random random;

        public SystemRandomProvider(Random random)
        {
            if (random == null) throw new ArgumentNullException(nameof(random));

            this.random = random;
        }

        public float NextFloat()
        {
            return (float)random.NextDouble();
        }
    }
}
