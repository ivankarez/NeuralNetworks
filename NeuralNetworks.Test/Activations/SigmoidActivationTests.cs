using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class SigmoidActivationTests
    {
        [TestCase(0f, 0.5f)]
        [TestCase(1f, 0.7310585786300049f)]
        [TestCase(-1f, 0.2689414213699951f)]
        [TestCase(2f, 0.8807970779778823f)]
        [TestCase(-2f, 0.11920292202211755f)]
        [TestCase(3f, 0.9525741268224334f)]
        [TestCase(-3f, 0.04742587317756678f)]
        public void TestApply(float input, float expectedOutput)
        {
            var output = new SigmoidActivation().Apply(input);
            output.Should().BeApproximately(expectedOutput, 0.0000001f);
        }
    }
}
