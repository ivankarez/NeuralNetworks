using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Values;
using NUnit.Framework;
using System.Linq;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class RecurrentLayerTests
    {
        [Test]
        public void TestUpdate_HappyPath()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), false);
            var parameters = new ValueStore(new float[] { 1, .5f, -1, -.5f });
            var values = new ValueStore();

            layer.Build(1, parameters, values);
            var result = layer.Update(ValueStoreRange.Of(1f));

            result.Count.Should().Be(2);
            result[0].Should().Be(1f);
            result[1].Should().Be(.5f);

            values.Values.Should().HaveCount(2);
        }

        [Test]
        public void TestUpdate_HappyPathWithBias()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), true);
            var parameters = new ValueStore(new float[] { 0f, 1.5f, 0f, 0.3f, 0f, 0f });
            var values = new ValueStore();

            layer.Build(1, parameters, values);
            var result = layer.Update(ValueStoreRange.Of(0f));

            result.Count.Should().Be(2);
            result[0].Should().Be(1.5f);
            result[1].Should().Be(0.3f);

            values.Values.Should().HaveCount(2);
        }

        [Test]
        public void TestUpdate_NotIdempotent()
        {
            var layer = new RecurrentLayer(2, new LinearActivation(), false);
            var parameters = new ValueStore(new float[] { 1, .5f, .1f, .1f });
            var values = new ValueStore();

            layer.Build(1, parameters, values);
            var result1 = layer.Update(ValueStoreRange.Of(0.5f)).ToArray();
            var result2 = layer.Update(ValueStoreRange.Of(0.5f)).ToArray();

            result1[0].Should().NotBe(result2[0]);
            result1[1].Should().NotBe(result2[1]);
            result2[0].Should().Be(0.55f);
            result2[1].Should().Be(0.275f);
        }

        [Test]
        public void TestUpdate_SameAsDenseIfNoRecurrentWeight()
        {
            var recurrentLayer = new RecurrentLayer(2, new LinearActivation(), false);
            var denseLayer = new DenseLayer(2, new LinearActivation(), false);

            var recurrentParameters = new ValueStore(new float[] { 1, .5f, 0f, 0f });
            var recurrentValues = new ValueStore();
            var denseParameters = new ValueStore(new float[] { 1, .5f });
            var denseVaues = new ValueStore();

            recurrentLayer.Build(1, recurrentParameters, recurrentValues);
            var recurrentResult = recurrentLayer.Update(ValueStoreRange.Of(1)).ToArray();

            denseLayer.Build(1, denseParameters, denseVaues);
            var denseResult = recurrentLayer.Update(ValueStoreRange.Of(1)).ToArray();

            recurrentResult.Should().BeEquivalentTo(denseResult);
        }
    }
}
