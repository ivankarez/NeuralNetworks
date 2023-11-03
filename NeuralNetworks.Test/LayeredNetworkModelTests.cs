using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;
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
            var model = NN.Models.Layered(new Size2D(125, 125),
                    NN.Layers.Conv2D((5, 5), useBias: false, kernelInitializer:NN.Initializers.GlorotNormal(randomProvider)),
                    NN.Layers.Pooling2D((10, 10), (10, 10)),
                    NN.Layers.Dense(12*12, useBias: false, kernelInitializer: NN.Initializers.GlorotUniform(randomProvider)),
                    NN.Layers.SimpleRecurrent(10, useBias: false, kernelInitializer: NN.Initializers.Normal(randomProvider: randomProvider)),
                    NN.Layers.Dense(3, kernelInitializer: NN.Initializers.Uniform(randomProvider: randomProvider), biasInitializer: NN.Initializers.Zeros())
                );

            var result = model.Feedforward(randomProvider.NextFloats(0, 1, inputMatrixSize * inputMatrixSize));

            result.Should().BeEquivalentTo(new float[] { 0.85433555f, 0.11379692f, 0.8347321f });
        }
    }
}
