using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.RandomGeneration
{
    public static class RandomProviderExtensions
    {
        public static float NextFloat(this IRandomProvider random, float max)
        {
            return NextFloat(random, 0f, max);
        }

        public static float NextFloat(this IRandomProvider random, float min, float max)
        {
            return random.NextFloat() * (max - min) + min;
        }

        public static float[] NextFloats(this IRandomProvider random, float max, int count)
        {
            return NextFloats(random, 0f, max, count);
        }

        public static float[] NextFloats(this IRandomProvider random, float min, float max, int count)
        {
            var result = new float[count];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = random.NextFloat(min, max);
            }

            return result;
        }

        public static float[] NextNormalFloats(this IRandomProvider random, float stdDev, int count)
        {
            return NextNormalFloats(random, 0f, stdDev, count);
        }

        public static float[] NextNormalFloats(this IRandomProvider random, float mean, float stdDev, int count)
        {
            var result = new float[count];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = random.NextNormalFloat(mean, stdDev);
            }

            return result;
        }

        public static float NextNormalFloat(this IRandomProvider rand, float stdDev)
        {
            return NextNormalFloat(rand, 0f, stdDev);
        }

        public static float NextNormalFloat(this IRandomProvider rand, float mean, float stdDev)
        {
            var u1 = rand.NextFloat();
            var u2 = rand.NextFloat();
            var randStdNormal = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));

            return mean + stdDev * randStdNormal;
        }

        public static int[] NextInts(this IRandomProvider random, int max, int count)
        {
            return NextInts(random, 0, max, count);
        }

        public static int[] NextInts(this IRandomProvider random, int min, int max, int count)
        {
            var result = new int[count];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = random.NextInt(min, max);
            }
            return result;
        }

        public static int NextInt(this IRandomProvider random, int max)
        {
            return NextInt(random, 0, max);
        }

        public static int NextInt(this IRandomProvider random, int min, int max)
        {
            return (int)random.NextFloat(min, max);
        }

        public static bool NextBool(this IRandomProvider random, float probability)
        {
            return random.NextFloat() < probability;
        }
    }
}
