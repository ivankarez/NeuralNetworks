using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Test.TestUtils;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    internal class DropoutLayerTests
    {
        [Test]
        public void Build_ShouldSetOutputSize()
        {
            var inputSize = NN.Size.Of(10);
            var layer = NN.Layers.Dropout(0.5f);

            layer.Build(inputSize);

            inputSize.TotalSize.Should().Be(layer.OutputSize.TotalSize);
        }

        [Test]
        public void Update_ShouldDropoutNodes()
        {
            var inputValues = new float[] { 1, 2, 3, 4, 5 };
            var layer = NN.Layers.Dropout(0.5f, new FakeRandomProvider(.4f, .6f, .6f, .4f, .4f));

            layer.Build(NN.Size.Of(inputValues.Length));
            var result = layer.Update(inputValues);

            result.Should().BeEquivalentTo(new float[] { 1, 0, 0, 4, 5 });
        }

        [Test]
        public void Update_ShouldDropoutNodesWithDifferentRate()
        {
            var inputValues = new float[] { 1, 2, 3, 4, 5 };
            var layer = NN.Layers.Dropout(0.3f, new FakeRandomProvider(.1f, .4f, .1f, .5f, .1f));

            layer.Build(NN.Size.Of(inputValues.Length));
            var result = layer.Update(inputValues);

            result.Should().BeEquivalentTo(new float[] { 1, 0, 3, 0, 5 });
        }
    }
}
