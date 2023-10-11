using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class RecurrentLayerTests
    {
        [Test]
        public void TestUpdate_SingleCall()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), false);
            layer.Build(1);
            layer.Parameters.Get2dVector("weights").Fill(new float[,] { { 1 }, { -1 } });
            layer.Parameters.Get1dVector("recurrentWeights").Fill(.5f, -.5f);
            var result = layer.Update(new float[] { 1f });

            result.Should().HaveCount(2);
            result[0].Should().Be(1f);
            result[1].Should().Be(-1f);

            layer.State.Get1dVector("nodeValues").Should().HaveCount(2);
        }

        [Test]
        public void TestUpdate_DoubleCall()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), false);
            layer.Build(1);
            layer.Parameters.Get2dVector("weights").Fill(new float[,] { { 1 }, { -1 } });
            layer.Parameters.Get1dVector("recurrentWeights").Fill(.5f, -.5f);

            layer.Update(new float[] { 1f });
            var result = layer.Update(new float[] { 1f });

            result.Should().HaveCount(2);
            result[0].Should().Be(1.5f);
            result[1].Should().Be(-.5f);

            layer.State.Get1dVector("nodeValues").Should().HaveCount(2);
        }

        [Test]
        public void TestUpdate_HappyPathWithBias()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), true);
            layer.Build(1);
            layer.Parameters.Get2dVector("weights").Fill(new float[,] { { 1 }, { -1 } });
            layer.Parameters.Get1dVector("recurrentWeights").Fill(.5f, -.5f);
            layer.Parameters.Get1dVector("biases").Fill(10, 10);

            var result = layer.Update(new float[] { 1f });

            result.Should().HaveCount(2);
            result[0].Should().Be(11f);
            result[1].Should().Be(9f);

            layer.State.Get1dVector("nodeValues").Should().HaveCount(2);
        }
    }
}
