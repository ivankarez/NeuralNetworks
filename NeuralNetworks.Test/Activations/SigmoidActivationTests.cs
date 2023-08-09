using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class SigmoidActivationTests
    {
        [TestCase(new float[] { 0f }, 0.5f)]
        [TestCase(new float[] { 1f }, 0.7310586f)]
        [TestCase(new float[] { -1f }, 0.2689414f)]
        [TestCase(new float[] { 2f }, 0.8807971f)]
        [TestCase(new float[] { -2f }, 0.1192029f)]
        [TestCase(new float[] { 0.5f }, 0.6224593f)]
        [TestCase(new float[] { -0.5f }, 0.3775407f)]
        [TestCase(new float[] { 1f, 1f }, 1.4621172f)]
        public void TestApply(float[] inputs, float expectedOutput)
        {
            var output = new SigmoidActivation().Apply(inputs);
            output.Should().BeApproximately(expectedOutput, 0.0000001f);
        }
    }
}
