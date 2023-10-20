using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test
{
    public class LayeredNetworkModelTests
    {
        [Test]
        public void TestFeedforward_HappyPath()
        {
            var model = CreateTestModel();
            Randomize(model);

            var result = model.Feedforward(new float[] { .5f });

            result.Should().HaveCount(2);
            result[0].Should().BeApproximately(-.1286f, 0.001f);
            result[1].Should().BeApproximately(.4030f, 0.001f);
        }

        private static LayeredNetworkModel CreateTestModel()
        {
            var activation = new LinearActivation();
            var initializer = new ZerosInitializer();
            var layer1 = new DenseLayer(3, activation, false, initializer, initializer);
            var layer2 = new DenseLayer(2, activation, false, initializer, initializer);
            return new LayeredNetworkModel(1, layer1, layer2);
        }

        private static void Randomize(LayeredNetworkModel model)
        {
            var random = new Random(0);
            foreach (var layer in model.Layers)
            {
                foreach (var vector1dName in layer.Parameters.Get1dVectorNames())
                {
                    var vector = layer.Parameters.Get1dVector(vector1dName);
                    for (int i = 0; i < vector.Length; i++)
                    {
                        vector[i] = (float)(random.NextDouble() * 2 - 1);
                    }
                }
                foreach (var vector2dName in layer.Parameters.Get2dVectorNames())
                {
                    var vector = layer.Parameters.Get2dVector(vector2dName);
                    for (int i = 0; i < vector.GetLength(0); i++)
                    {
                        for (int j = 0; j < vector.GetLength(1); j++)
                        {
                            vector[i, j] = (float)(random.NextDouble() * 2 - 1);
                        }
                    }
                }
            }
        }
    }
}
