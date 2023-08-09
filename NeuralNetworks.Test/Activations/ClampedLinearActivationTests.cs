using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class ClampedLinearActivationTests
    {
        [TestCase(-1, 1, new float[] { 0f }, 0f)]
        [TestCase(-1, 1, new float[] { 1f }, 1f)]
        [TestCase(-1, 1, new float[] { -1f }, -1f)]
        [TestCase(-1, 1, new float[] { -1f, 1f }, 0f)]
        [TestCase(-1, 1, new float[] { -12f, 10f }, -1f)]
        [TestCase(-1, 1.5f, new float[] { 12f, -10f }, 1.5f)]
        [TestCase(1, 1.5f, new float[] { 0f }, 1f)]
        public void TestApply(float min, float max, float[] inputs, float expectedOutput)
        {
            var output = new ClampedLinearActivation(min, max).Apply(inputs);
            output.Should().Be(expectedOutput);
        }
    }
}