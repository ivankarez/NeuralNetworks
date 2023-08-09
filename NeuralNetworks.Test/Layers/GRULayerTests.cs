using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Test.TestUtils;
using Ivankarez.NeuralNetworks.Values;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class GRULayerTests
    {
        [Test]
        public void TestUpdate_HappyPath()
        {
            var layer = new GRULayer(2, new SigmoidActivation(), new SigmoidActivation(), true);
            var parameters = RandomTestUtils.CreateRandomValueStore(20, 0);
            layer.Build(1, parameters, new ValueStore());

            var result = layer.Update(ValueStoreRange.Of(2f));

            result.Should().HaveCount(2);
            result[0].Should().BeApproximately(-1.744814f, 0.000001f);
            result[1].Should().BeApproximately(-1.7011594f, 0.000001f);
        }

        [Test]
        public void TestUpdate_NotIdempotent()
        {
            var layer = new GRULayer(2, new SigmoidActivation(), new SigmoidActivation(), true);
            var parameters = RandomTestUtils.CreateRandomValueStore(20, 0);
            layer.Build(1, parameters, new ValueStore());

            var result = layer.Update(ValueStoreRange.Of(2f));
            result.Should().HaveCount(2);
            result[0].Should().BeApproximately(-1.744814f, 0.000001f);
            result[1].Should().BeApproximately(-1.7011594f, 0.000001f);

            var result2 = layer.Update(ValueStoreRange.Of(2f));
            result2.Should().HaveCount(2);
            result2[0].Should().BeApproximately(-5.9228706f, 0.000001f);
            result2[1].Should().BeApproximately(-6.8502493f, 0.000001f);
        }

        [Test]
        public void TestUpdate_WithoutBias()
        {
            var layer = new GRULayer(2, new SigmoidActivation(), new SigmoidActivation(), false);
            var parameters = RandomTestUtils.CreateRandomValueStore(20, 0);
            layer.Build(1, parameters, new ValueStore());

            var result = layer.Update(ValueStoreRange.Of(2f));

            result.Should().HaveCount(2);
            result[0].Should().BeApproximately(-0.41059923f, 0.000001f);
            result[1].Should().BeApproximately(-0.3175379f, 0.000001f);
        }
    }
}
