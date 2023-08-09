using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Test.TestUtils
{
    public static class RandomTestUtils
    {
        public static ValueStore CreateRandomValueStore(int size, int seed)
        {
            var random = new Random(seed);
            var values = new float[size];
            for (int i = 0; i < size; i++)
            {
                values[i] = (float)random.NextDouble();
            }
            return new ValueStore(values);
        }
    }
}
