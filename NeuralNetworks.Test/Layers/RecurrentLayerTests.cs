using FluentAssertions;
using NeuralNetworks.Activations;
using NeuralNetworks.Layers;
using NUnit.Framework;

namespace NeuralNetworks.Test.Layers
{
    public class RecurrentLayerTests
    {
        [Test]
        public void TestUpdate_HappyPath()
        {
            var layer = new RecurrentLayer(2, new LinearActivation());
            var parameters = new ModelParameters(new float[] { 1, -1, .5f, -.5f });
            var values = new ModelParameters();

            layer.Build(1, parameters, values);
            var result = layer.Update(ParameterRange.Of(1f));

            result.Size.Should().Be(2);
            result[0].Should().Be(1f);
            result[1].Should().Be(.5f);

            values.Values.Should().HaveCount(2);
        }

        [Test]
        public void TestUpdate_NotIdempotent()
        {
            var layer = new RecurrentLayer(2, new LinearActivation());
            var parameters = new ModelParameters(new float[] { 1, .1f, .5f, .1f });
            var values = new ModelParameters();

            layer.Build(1, parameters, values);
            var result1 = layer.Update(ParameterRange.Of(0.5f)).ToArray();
            var result2 = layer.Update(ParameterRange.Of(0.5f)).ToArray();

            result1[0].Should().NotBe(result2[0]);
            result1[1].Should().NotBe(result2[1]);
            result2[0].Should().Be(0.55f);
            result2[1].Should().Be(0.275f);
        }

        [Test]
        public void TestUpdate_SameAsDenseIfNoRecurrentWeight()
        {
            var recurrentLayer = new RecurrentLayer(2, new LinearActivation());
            var denseLayer = new DenseLayer(2, new LinearActivation(), false);

            var recurrentParameters = new ModelParameters(new float[] { 1, 0f, .5f, 0f });
            var recurrentValues = new ModelParameters();
            var denseParameters = new ModelParameters(new float[] { 1, .5f });
            var denseVaues = new ModelParameters();

            recurrentLayer.Build(1, recurrentParameters, recurrentValues);
            var recurrentResult = recurrentLayer.Update(ParameterRange.Of(1)).ToArray();

            denseLayer.Build(1, denseParameters, denseVaues);
            var denseResult = recurrentLayer.Update(ParameterRange.Of(1)).ToArray();

            recurrentResult.Should().BeEquivalentTo(denseResult);
        }
    }
}
