using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.RandomGeneration;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test
{
    public class LayeredNetworkModelTests
    {
        [Test]
        public void TestFeedforward_HappyPath()
        {
            var inputMatrixSize = 125;
            var randomProvider = NN.Random.System(new Random(0));
            var model = NN.Models.Layered(125 * 125,
                    NN.Layers.Conv2D(125, 125, 5, 5, 1, 1, useBias: false, kernelInitializer:NN.Initializers.GlorotNormal(randomProvider)),
                    NN.Layers.Pooling2D(121, 121, 10, 10, 10, 10),
                    NN.Layers.Dense(12*12, useBias: false, kernelInitializer: NN.Initializers.GlorotUniform(randomProvider)),
                    NN.Layers.SimpleRecurrent(10, useBias: false, kernelInitializer: NN.Initializers.Normal(randomProvider: randomProvider)),
                    NN.Layers.Dense(3, kernelInitializer: NN.Initializers.Uniform(randomProvider: randomProvider), biasInitializer: NN.Initializers.Zeros())
                );

            var result = model.Feedforward(randomProvider.NextFloats(0, 1, inputMatrixSize * inputMatrixSize));

            result.Should().BeEquivalentTo(new float[] { 0.8542598f, 0.113805376f, 0.8348296f });
        }
    }
}
