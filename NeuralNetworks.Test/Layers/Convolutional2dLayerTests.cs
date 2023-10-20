using FluentAssertions;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Test.TestUtils;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class Convolutional2dLayerTests
    {
        [Test]
        public void Build_WithInvalidInputSize_ThrowsArgumentException()
        {
            var layer = new Convolutional2dLayer(3, 3, 2, 2, 1, 1);
            Action testAction = () => layer.Build(4);
            testAction.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Build_WithValidInputSize_SetsNodeCount()
        {
            var layer = new Convolutional2dLayer(3, 3, 2, 2, 1, 1);
            layer.Build(9);
            layer.NodeCount.Should().Be(4);
        }

        [Test]
        public void Build_WithValidInputSize_SetsNodeValues()
        {
            var layer = new Convolutional2dLayer(3, 3, 2, 2, 1, 1);
            layer.Build(9);
            layer.State.Get1dVector("nodeValues").Should().HaveCount(4);
        }

        [Test]
        public void Update_WithZeros()
        {
            var layer = new Convolutional2dLayer(3, 3, 2, 2, 1, 1);
            layer.Build(9);
            var output = layer.Update(new float[9]);
            output.Should().OnlyContain(v => v == 0);
        }

        [Test]
        public void Update_WithData()
        {
            var layer = new Convolutional2dLayer(3, 3, 2, 2, 1, 1);
            layer.Build(9);

            layer.Parameters.Get2dVector("filter").Fill(new float[,] { { -1, 2, }, { -2, 3 } });
            var input = new float[] { 1, 2, 3, -1, -2, -3, 1, 2, 3 };
            var output = layer.Update(input);
            var expected = new float[] { -1, -1, 1, 1 };
            output.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void Update_WithData2()
        {
            var layer = new Convolutional2dLayer(3, 3, 3, 3, 1, 1);
            layer.Build(9);

            layer.Parameters.Get2dVector("filter").Fill(new float[,] { { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } });
            var input = new float[] { 1, 2, 3, -1, -2, -3, 1, 2, 3 };
            var output = layer.Update(input);
            var expected = new float[] { 12 };
            output.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void Update_WithData_LargerInput()
        {
            var layer = new Convolutional2dLayer(90, 90, 11, 11, 5, 5);
            layer.Build(90 * 90);

            var filter = RandomTestUtils.CreateRandomFloatMatrix(11, 11, 0);
            layer.Parameters.Get2dVector("filter").Fill(filter);
            var output = layer.Update(new float[90 * 90]);
            var expected = new float[] { 12 };
            output.Should().OnlyContain(v => v == 0);
        }
    }
}
