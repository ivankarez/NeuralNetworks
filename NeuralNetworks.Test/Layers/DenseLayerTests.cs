using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class DenseLayerTests
    {
        [Test]
        public void TestUpdate_HappyPath()
        {
            var layer = new DenseLayer(2, new LinearActivation(), false);
            layer.Build(1);
            layer.Parameters.Get2dVector("weights").Fill( new float[,] { { -1f }, { 2.3f } });

            var result = layer.Update(new float[] { 2f });

            result.Should().HaveCount(2);
            result[0].Should().Be(-2f);
            result[1].Should().Be(4.6f);
        }

        [Test]
        public void TestUpdate_HappyPath2()
        {
            var layer = new DenseLayer(3, new LinearActivation(), false);
            layer.Build(2);
            layer.Parameters.Get2dVector("weights").Fill(new float[,] { { -1f, 2.3f }, { 1.34f, .5f }, { -.34f, .2f } });

            var result = layer.Update(new float[] { 2f, -.5f });

            result.Should().HaveCount(3);
            result[0].Should().BeApproximately(-3.15f, .01f);
            result[1].Should().BeApproximately(2.43f, .01f);
            result[2].Should().BeApproximately(-.78f, .01f);
        }

        [Test]
        public void TestUpdate_WithBias()
        {
            var layer = new DenseLayer(2, new LinearActivation(), true);
            layer.Build(1);
            layer.Parameters.Get1dVector("biases").Fill(.5f, -.23f);

            var result = layer.Update(new float[] { 2f });

            result.Should().HaveCount(2);
            result[0].Should().Be(.5f);
            result[1].Should().Be(-.23f);
        }
    }
}
