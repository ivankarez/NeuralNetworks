using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test
{
    public class LayeredNetworkModelFlatParametersExtensionsTests
    {
        [Test]
        public void Test_GetParametersFlat_HappyPath()
        {
            var model = NN.Models.Layered(2, NN.Layers.Dense(2), NN.Layers.Dense(3));
            var parameters = model.GetParametersFlat();

            parameters.Should().HaveCount(2 * 2 + 2 * 3 + 5); // Layer weights and 5 biases
        }

        [Test]
        public void Test_SetParametersFlat_HappyPath()
        {
            var layer = NN.Layers.Dense(2);
            var model = NN.Models.Layered(2, layer);
            model.SetParametersFlat(new float[] { 3, 3, 3, 3, 3, 3 });

            layer.Parameters.Get1dVector("biases").Should().OnlyContain(v => v == 3f);
            var weights = layer.Parameters.Get2dVector("weights");
            foreach (var weight in weights)
            {
                weight.Should().Be(3);
            }

            layer.State.Get1dVector("nodeValues").Should().OnlyContain(v => v == 0);
        }

        [Test]
        public void Test_CountParameters_HappyPath()
        {
            var model = NN.Models.Layered(2, NN.Layers.Dense(2), NN.Layers.Dense(3));
            model.CountParameters().Should().Be(2 * 2 + 2 * 3 + 5); // Layer weights and 5 biases
        }
    }
}
