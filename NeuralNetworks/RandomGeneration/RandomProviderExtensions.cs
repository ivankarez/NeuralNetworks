using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.RandomGeneration
{
    public static class RandomProviderExtensions
    {
        public static float NextFloat(this IRandomProvider random, float min, float max)
        {
            return random.NextFloat() * (max - min) + min;
        }

        public static float NextNormalFloat(this IRandomProvider rand, float mean, float stdDev)
        {
            var u1 = rand.NextFloat();
            var u2 = rand.NextFloat();
            var randStdNormal = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));

            return mean + stdDev * randStdNormal;
        }
    }
}
