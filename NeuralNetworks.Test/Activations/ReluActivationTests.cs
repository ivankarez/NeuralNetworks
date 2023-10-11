using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class ReluActivationTests
    {
        [TestCase(0f, 0f)]
        [TestCase(.5f, .5f)]
        [TestCase(1f, 1f)]
        [TestCase(10f, 10f)]
        [TestCase(-0.01f, 0f)]
        [TestCase(-10f, 0f)]
        public void TestApply(float input, float expectedOutput)
        {
            var output = new ReluActivation().Apply(input);
            output.Should().Be(expectedOutput);
        }
    }
}
