using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class LeakyReluActivationTests
    {
        [TestCase(0f, 0f)]
        [TestCase(1f, 1f)]
        [TestCase(2f, 2f)]
        [TestCase(-1f, -0.1f)]
        [TestCase(-2f, -0.2f)]
        public void TestApply(float input, float expectedOutput)
        {
            var output = new LeakyReluActivation(.1f).Apply(input);
            output.Should().Be(expectedOutput);
        }
    }
}
