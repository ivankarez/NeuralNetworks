using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class LinearActivationTests
    {
        [Test]
        public void TestApply_HappyPath()
        {
            var activation = new LinearActivation();
            var inputs = new float[] { 1.5f, -1.2f, 1f };

            var output = activation.Apply(inputs);

            output.Should().Be(1.3f);
        }
    }
}