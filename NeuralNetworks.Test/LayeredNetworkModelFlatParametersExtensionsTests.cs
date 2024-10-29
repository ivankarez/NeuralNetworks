using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test
{
    public class LayeredNetworkModelFlatParametersExtensionsTests
    {
        [Test]
        public void Test_GetParametersFlat_HappyPath()
        {
            var model = NN.Models.Layered(2, NN.Layers.Dense(2), NN.Layers.Dense(3));

            throw new Exception("Test not implemented yet");
        }

        [Test]
        public void Test_SetParametersFlat_HappyPath()
        {
            var layer = NN.Layers.Dense(2);
            var model = NN.Models.Layered(2, layer);
            model.SetParametersFlat(new float[] { 3, 3, 3, 3, 3, 3 });

            throw new Exception("Test not implemented yet");
        }

        [Test]
        public void Test_CountParameters_HappyPath()
        {
            var model = NN.Models.Layered(2, NN.Layers.Dense(2), NN.Layers.Dense(3));
            model.CountParameters().Should().Be(2 * 2 + 2 * 3 + 5); // Layer weights and 5 biases
        }
    }
}
