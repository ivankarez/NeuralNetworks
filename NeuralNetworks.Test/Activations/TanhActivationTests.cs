using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class TanhActivationTests
    {
        [Test]
        public void TestApply_HappyPath()
        {
            var activation = new TanhActivation();
            var result = activation.Apply(0.5f);
            result.Should().BeApproximately(0.462117157f, 0.0001f);
        }

        [Test]
        public void TestApply_NegativeInput()
        {
            var activation = new TanhActivation();
            var result = activation.Apply(-0.5f);
            result.Should().BeApproximately(-0.462117157f, 0.0001f);
        }

        [Test]
        public void TestApply_ZeroInput()
        {
            var activation = new TanhActivation();
            var result = activation.Apply(0f);
            result.Should().BeApproximately(0f, 0.0001f);
        }
    }
}
