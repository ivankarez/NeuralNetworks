using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
using Ivankarez.NeuralNetworks.Test.TestUtils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class Convolutional2dLayerTests
    {
        [Test]
        public void Build_WithInvalidInputSize_ThrowsArgumentException()
        {
            var layer = NN.Layers.Conv2D((2, 2));
            Action testAction = () => layer.Build(NN.Size.Of(1));
            testAction.Should().Throw<ArgumentException>();
        }

        [Test]
        public void Build_WithValidInputSize_SetsNodeCount()
        {
            var layer = NN.Layers.Conv2D((2, 2), useBias: false);
            layer.Build(NN.Size.Of(3, 3));
            layer.OutputSize.Should().Be(NN.Size.Of(2, 2));
        }

        [Test]
        public void Build_WithValidInputSize_UseBias()
        {
            var layer = NN.Layers.Conv2D((2, 2));
            layer.Build(NN.Size.Of(3, 3));
            layer.OutputSize.Should().Be(NN.Size.Of(2, 2));
            layer.Biases.Should().HaveCount(4);
        }

        [Test]
        public void Build_UseInitializer()
        {
            var layer = NN.Layers.Conv2D((2, 2), kernelInitializer: new ConstantInitializer(3f), biasInitializer: new ConstantInitializer(2));
            layer.Build(NN.Size.Of(3, 3));
            layer.Filter.Should().BeEquivalentTo(new float[][] { new float[] { 3, 3 }, new float[] { 3, 3 } });
            layer.Biases.Should().OnlyContain(v => v == 2);
        }

        [Test]
        public void Build_WithValidInputSize_SetsNodeValues()
        {
            var layer = NN.Layers.Conv2D((2, 2), useBias: false);
            layer.Build(NN.Size.Of(3, 3));

            var result = layer.Update(new float[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 });

            result.Should().HaveCount(4);
        }

        [Test]
        public void Update_WithZeros()
        {
            var layer = NN.Layers.Conv2D((2, 2), useBias: false);
            layer.Build(NN.Size.Of(3, 3));
            var output = layer.Update(new float[9]);
            output.Should().OnlyContain(v => v == 0);
        }

        [Test]
        public void Update_WithData()
        {
            var layer = NN.Layers.Conv2D((2, 2), useBias: false);
            layer.Build(NN.Size.Of(3, 3));

            layer.Filter = new float[][] { new float[] { -1, 2, }, new float[] { -2, 3 } };
            var input = new float[] { 1, 2, 3, -1, -2, -3, 1, 2, 3 };
            var output = layer.Update(input);
            var expected = new float[] { -1, -1, 1, 1 };
            output.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void Update_WithData2()
        {
            var layer = NN.Layers.Conv2D((3, 3), useBias: false);
            layer.Build(NN.Size.Of(3, 3));

            layer.Filter = new float[][] { new float[] { 1, 1, 1 }, new float[] { 2, 2, 2 }, new float[] { 3, 3, 3 } };
            var input = new float[] { 1, 2, 3, -1, -2, -3, 1, 2, 3 };
            var output = layer.Update(input);
            var expected = new float[] { 12 };
            output.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void Update_WithData_LargerInput()
        {
            var layer = NN.Layers.Conv2D((11, 11), (5, 5), false);
            layer.Build(NN.Size.Of(90, 90));

            var filter = RandomTestUtils.CreateRandomFloatMatrix2(11, 11, 0);
            layer.Filter = filter;
            var output = layer.Update(new float[90 * 90]);
            var expected = new float[] { 12 };
            output.Should().OnlyContain(v => v == 0);
        }

        [Test]
        public void Update_WithBias()
        {
            var layer = NN.Layers.Conv2D((2, 2), kernelInitializer: new ConstantInitializer(1), biasInitializer: new ConstantInitializer(2));
            layer.Build(NN.Size.Of(3, 3));

            var output = layer.Update(new float[] {
                1, 2, 3,
                -1, -2, -3,
                1, -1, 0 });
            output.Should().BeEquivalentTo(new float[] { 2, 2, -1, -4 });
        }
    }
}
