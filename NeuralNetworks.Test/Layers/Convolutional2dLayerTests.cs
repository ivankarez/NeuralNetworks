using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
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
            layer.Parameters.Get1dVector("biases").Should().HaveCount(4);
        }

        [Test]
        public void Build_UseInitializer()
        {
            var layer = NN.Layers.Conv2D((2, 2), kernelInitializer: new ConstantInitializer(3f), biasInitializer: new ConstantInitializer(2));
            layer.Build(NN.Size.Of(3, 3));
            layer.Parameters.Get2dVector("filter").Should().BeEquivalentTo(new float[,] { { 3, 3 }, { 3, 3 } });
            layer.Parameters.Get1dVector("biases").Should().OnlyContain(v => v == 2);
        }

        [Test]
        public void Build_WithValidInputSize_SetsNodeValues()
        {
            var layer = NN.Layers.Conv2D((2, 2), useBias: false);
            layer.Build(NN.Size.Of(3, 3));
            layer.State.Get1dVector("nodeValues").Should().HaveCount(4);
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

            layer.Parameters.Get2dVector("filter").Fill(new float[,] { { -1, 2, }, { -2, 3 } });
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

            layer.Parameters.Get2dVector("filter").Fill(new float[,] { { 1, 1, 1 }, { 2, 2, 2 }, { 3, 3, 3 } });
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

            var filter = RandomTestUtils.CreateRandomFloatMatrix(11, 11, 0);
            layer.Parameters.Get2dVector("filter").Fill(filter);
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
