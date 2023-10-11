using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class ClampedLinearActivationTests
    {
        [TestCase(-1, 1, 0f, 0f)]
        [TestCase(-1, 1, 1f, 1f)]
        [TestCase(-1, 1, -1f, -1f)]
        [TestCase(-1, 1, 2f, 1f)]
        [TestCase(-1, 1, -2f -1f, -1f)]
        public void TestApply(float min, float max, float input, float expectedOutput)
        {
            var output = new ClampedLinearActivation(min, max).Apply(input);
            output.Should().Be(expectedOutput);
        }
    }
}