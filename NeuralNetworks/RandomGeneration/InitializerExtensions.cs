using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.RandomGeneration
{
    public static class InitializerExtensions
    {
        public static float[] GenerateValues(this IInitializer initializer, int fanIn, int fanOut, int count)
        {
            var values = new float[count];
            for (int i = 0; i < count; i++)
            {
                values[i] = initializer.GenerateValue(fanIn, fanOut);
            }

            return values;
        }

        public static float[,] GenerateValues2d(this IInitializer initializer, int fanIn, int fanOut, int width, int height)
        {
            var values = new float[width, height];
            for (int x = 0; x < width; x++)
            {
                for (int y = 0; y < height; y++)
                {
                    values[x, y] = initializer.GenerateValue(fanIn, fanOut);
                }
            }

            return values;
        }
    }
}
