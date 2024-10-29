using FluentAssertions;
using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class RecurrentLayerTests
    {
        private static readonly IInitializer defaultInitializer = new ZerosInitializer();

        [Test]
        public void TestUpdate_SingleCall()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), false, defaultInitializer, defaultInitializer, defaultInitializer);
            layer.Build(NN.Size.Of(1));
            layer.Weights = new float[,] { { 1 }, { -1 } };
            layer.RecurrentWeights = new float[] { .5f, -.5f };
            var result = layer.Update(new float[] { 1f });

            result.Should().HaveCount(2);
            result[0].Should().Be(1f);
            result[1].Should().Be(-1f);

            layer.Output.Should().BeSameAs(result);
        }

        [Test]
        public void TestUpdate_DoubleCall()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), false, defaultInitializer, defaultInitializer, defaultInitializer);
            layer.Build(NN.Size.Of(1));
            layer.Weights = new float[,] { { 1 }, { -1 } };
            layer.RecurrentWeights = new float[] { .5f, -.5f };

            layer.Update(new float[] { 1f });
            var result = layer.Update(new float[] { 1f });

            result.Should().HaveCount(2);
            result[0].Should().Be(1.5f);
            result[1].Should().Be(-.5f);

            layer.Output.Should().BeSameAs(result);
        }

        [Test]
        public void TestUpdate_HappyPathWithBias()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), true, defaultInitializer, defaultInitializer, defaultInitializer);
            layer.Build(NN.Size.Of(1));
            layer.Weights = new float[,] { { 1 }, { -1 } };
            layer.RecurrentWeights = new float[] { .5f, -.5f };
            layer.Biases = new float[] { 10, 10 };

            var result = layer.Update(new float[] { 1f });

            result.Should().HaveCount(2);
            result[0].Should().Be(11f);
            result[1].Should().Be(9f);

            layer.Output.Should().BeSameAs(result);
        }

        [Test]
        public void TestBuild_UseInitializers()
        {
            var kernelInitializer = new ConstantInitializer(1);
            var recurrentInitializer = new ConstantInitializer(2);
            var biasInitializer = new ConstantInitializer(3);

            var layer = new RecurrentLayer(2, new LinearActivation(), true, kernelInitializer, biasInitializer, recurrentInitializer);
            layer.Build(NN.Size.Of(1));

            layer.Weights.Should().BeEquivalentTo(new float[,] { { 1 }, { 1 } });
            layer.RecurrentWeights.Should().BeEquivalentTo(new float[] { 2, 2 });
            layer.Biases.Should().BeEquivalentTo(new float[] { 3, 3 });
        }
    }
}
