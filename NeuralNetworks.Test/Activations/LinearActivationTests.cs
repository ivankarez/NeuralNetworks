using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class LinearActivationTests
    {
        [TestCase( 0f, 0f)]
        [TestCase(1f, 1f)]
        [TestCase(0.5f, 0.5f)]
        [TestCase(-1f, -1f)]
        [TestCase(-0.5f, -0.5f)]
        [TestCase(2f, 2f)]
        [TestCase(-2f, -2f)]
        public void TestApply(float input, float expectedOutput)
        {
            var output = new LinearActivation().Apply(input);
            output.Should().Be(expectedOutput);
        }
    }
}