using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class DenseLayerTests
    {
        [Test]
        public void TestUpdate_HappyPath()
        {
            var initializer = new ZerosInitializer();
            var layer = new DenseLayer(2, new LinearActivation(), false, initializer, initializer);
            layer.Build(NN.Size.Of(1));
            layer.Parameters.Get2dVector("weights").Fill(new float[,] { { -1f }, { 2.3f } });

            var result = layer.Update(new float[] { 2f });

            result.Should().HaveCount(2);
            result[0].Should().Be(-2f);
            result[1].Should().Be(4.6f);
        }

        [Test]
        public void TestUpdate_HappyPath2()
        {
            var initializer = new ZerosInitializer();
            var layer = new DenseLayer(3, new LinearActivation(), false, initializer, initializer);
            layer.Build(NN.Size.Of(2));
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
            var initializer = new ZerosInitializer();
            var layer = new DenseLayer(2, new LinearActivation(), true, initializer, initializer);
            layer.Build(NN.Size.Of(1));
            layer.Parameters.Get1dVector("biases").Fill(.5f, -.23f);

            var result = layer.Update(new float[] { 2f });

            result.Should().HaveCount(2);
            result[0].Should().Be(.5f);
            result[1].Should().Be(-.23f);
        }

        [Test]
        public void TestBuild_UseInitializers()
        {
            var kernelInitializer = new ConstantInitializer(1);
            var biasInitializer = new ConstantInitializer(2);
            var layer = new DenseLayer(2, new LinearActivation(), true, kernelInitializer, biasInitializer);
            layer.Build(NN.Size.Of(10));

            var weights = layer.Parameters.Get2dVector("weights");
            for (int x = 0; x < weights.GetLength(0); x++)
            {
                for (int y = 0; y < weights.GetLength(1); y++) {
                    weights[x, y].Should().Be(1);
                }
            }

            layer.Parameters.Get1dVector("biases").Should().OnlyContain(v => v == 2);
        }
    }
}
