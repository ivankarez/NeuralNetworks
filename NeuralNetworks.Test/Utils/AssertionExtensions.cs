using FluentAssertions;
using FluentAssertions.Collections;

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

        public static AndConstraint<GenericCollectionAssertions<float[]>> OnlyContainNumber(this GenericCollectionAssertions<float[]> assertion, float value)
        {
            return assertion.AllSatisfy(v => v.Should().OnlyContain(n => n==value));
        }

        public static AndConstraint<GenericCollectionAssertions<float>> OnlyContainNumber(this GenericCollectionAssertions<float> assertion, float value)
        {
            return assertion.OnlyContain(n => n == value);
        }

        public static void ShouldOnlyContain(this float[] values, float value)
        {
            for (var i = 0; i < values.Length; i++)
            {
                values[i].Should().Be(value);
            }
        }
    }
}
