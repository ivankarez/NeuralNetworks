using FluentAssertions;
using NeuralNetworks.Activations;
using NUnit.Framework;

namespace NeuralNetworks.Test.Activations
{
    public class ClampedLinearActivationTests
    {
        [Test]
        public void TestApply_ClampMin()
        {
            var activation = new ClampedLinearActivation(-1, 1);
            var inputs = new float[] { -1.5f, -1.2f, 1f };

            var output = activation.Apply(inputs);

            output.Should().Be(-1);
        }

        [Test]
        public void TestApply_ClampMax()
        {
            var activation = new ClampedLinearActivation(-1, 1);
            var inputs = new float[] { 1.5f, 1.2f, 1f };

            var output = activation.Apply(inputs);

            output.Should().Be(1);
        }
    }
}