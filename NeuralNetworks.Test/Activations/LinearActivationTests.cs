using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class LinearActivationTests
    {
        [TestCase(new float[] { 0f }, 0f)]
        [TestCase(new float[] { 1f }, 1f)]
        [TestCase(new float[] { -1f }, -1f)]
        [TestCase(new float[] { -1f, 1f }, 0f)]
        [TestCase(new float[] { -12f, 10f }, -2f)]
        public void TestApply(float[] inputs, float expectedOutput)
        {
            var output = new LinearActivation().Apply(inputs);
            output.Should().Be(expectedOutput);
        }
    }
}