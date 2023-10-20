using System;

namespace Ivankarez.NeuralNetworks.Test.TestUtils
{
    public static class RandomTestUtils
    {
        public static float[] CreateRandomFloatArray(int size, int seed)
        {
            var random = new Random(seed);
            var values = new float[size];
            for (int i = 0; i < size; i++)
            {
                values[i] = (float)random.NextDouble();
            }
            return values;
        }

        public static float[,] CreateRandomFloatMatrix(int width, int height, int seed)
        {
            var random = new Random(seed);
            var values = new float[width, height];
            for (int i = 0; i < width; i++)
            {
                for (int j = 0; j < height; j++)
                {
                    values[i, j] = (float)random.NextDouble();
                }
            }
            return values;
        }
    }
}
