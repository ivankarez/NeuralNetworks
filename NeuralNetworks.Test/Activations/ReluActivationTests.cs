using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class ReluActivationTests
    {
        [TestCase(new float[] { 0f }, 0f)]
        [TestCase(new float[] { .5f }, .5f)]
        [TestCase(new float[] { 1f }, 1f)]
        [TestCase(new float[] { 10f }, 10f)]
        [TestCase(new float[] { -0.01f }, 0f)]
        [TestCase(new float[] { -10f }, 0f)]
        [TestCase(new float[] { 2f, 1f, -10f }, 3f)]
        public void TestApply(float[] inputs, float expectedOutput)
        {
            var output = new ReluActivation().Apply(inputs);
            output.Should().Be(expectedOutput);
        }
    }
}
