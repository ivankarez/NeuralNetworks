using FluentAssertions;
using NeuralNetworks.Activations;
using NeuralNetworks.Layers;
using NUnit.Framework;
using System;

namespace NeuralNetworks.Test
{
    public class LayeredNetworkModelTests
    {
        [Test]
        public void TestPredict_HappyPath()
        {
            var model = CreateTestModel();
            Randomize(model);

            var result = model.Predict(new float[] { .5f });

            result.Should().HaveCount(2);
            result[0].Should().BeApproximately(-.1286f, 0.001f);
            result[1].Should().BeApproximately(.4030f, 0.001f);
        }

        [Test]
        public void TestPredict_CopyParameters()
        {
            var model1 = CreateTestModel();
            Randomize(model1);

            var model2 = CreateTestModel();
            for (int i = 0; i < model1.Parameters.Count; i++)
            {
                model2.Parameters.Values[i] = model1.Parameters.Values[i];
            }

            var inputs = new float[] { .12f };
            var result1 = model1.Predict(inputs);
            var result2 = model2.Predict(inputs);

            result2.Should().BeEquivalentTo(result1);
        }

        private static LayeredNetworkModel CreateTestModel()
        {
            var activation = new LinearActivation();
            var layer1 = new DenseLayer(3, activation, false);
            var layer2 = new DenseLayer(2, activation, false);
            return new LayeredNetworkModel(1, layer1, layer2);
        }

        private static void Randomize(LayeredNetworkModel model)
        {
            var random = new Random(0);
            for (int i = 0; i < model.Parameters.Count; i++)
            {
                model.Parameters.Values[i] = (float)(random.NextDouble() * 2 - 1);
            }
        }
    }
}
