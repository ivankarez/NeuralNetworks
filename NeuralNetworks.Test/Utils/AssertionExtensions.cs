using FluentAssertions;
using Ivankarez.NeuralNetworks.Values;

namespace Ivankarez.NeuralNetworks.Test.Utils
{
    public static class AssertionExtensions
    {
        public static void ShouldOnlyContain(this float[,] values, float value)
        {
            for (var i = 0; i < values.GetLength(0); i++)
            {
                for (var j = 0; j < values.GetLength(1); j++)
                {
                    values[i, j].Should().Be(value);
                }
            }
        }

        public static void ShouldOnlyContain(this float[] values, float value)
        {
            for (var i = 0; i < values.Length; i++)
            {
                values[i].Should().Be(value);
            }
        }

        public static float[] ShouldContain1D(this NamedVectors<float> namedVectors, string vectorName)
        {
            namedVectors.Get1dVectorNames().Should().Contain(vectorName);
            return namedVectors.Get1dVector(vectorName);
        }

        public static float[,] ShouldContain2D(this NamedVectors<float> namedVectors, string vectorName)
        {
            namedVectors.Get2dVectorNames().Should().Contain(vectorName);
            return namedVectors.Get2dVector(vectorName);
        }

        public static void ShouldNotContain1D(this NamedVectors<float> namedVectors, string vectorName)
        {
            namedVectors.Get1dVectorNames().Should().NotContain(vectorName);
        }
    }
}
