using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class PoolingLayerTests
    {
        [Test]
        public void TestUpdate_MaxPooling()
        {
            var layer = new PoolingLayer(3, 3, PoolingType.Max);
            layer.Build(NN.Size.Of(9));
            layer.OutputSize.Should().Be(NN.Size.Of(3));

            var result = layer.Update(new float[] { 1f, 2f, 3f, 5f, 1f, 1f, -1f, -2f, -1f });

            result.Should().HaveCount(3);
            result[0].Should().Be(3f);
            result[1].Should().Be(5f);
            result[2].Should().Be(-1f);
        }

        [Test]
        public void TestUpdate_MinPooling()
        {
            var layer = new PoolingLayer(3, 3, PoolingType.Min);
            layer.Build(NN.Size.Of(9));
            layer.OutputSize.Should().Be(NN.Size.Of(3));

            var result = layer.Update(new float[] { 1f, 2f, 3f, 5f, 1f, 0f, -1f, -2f, -1f });

            result.Should().HaveCount(3);
            result[0].Should().Be(1f);
            result[1].Should().Be(0f);
            result[2].Should().Be(-2f);
        }

        [Test]
        public void TestUpdate_SumPooling()
        {
            var layer = new PoolingLayer(3, 3, PoolingType.Sum);
            layer.Build(NN.Size.Of(9));
            layer.OutputSize.Should().Be(NN.Size.Of(3));

            var result = layer.Update(new float[] { 1f, 2f, 3f, 5f, 1f, -2f, -1f, -2f, -1f });

            result.Should().HaveCount(3);
            result[0].Should().Be(6f);
            result[1].Should().Be(4f);
            result[2].Should().Be(-4f);
        }

        [Test]
        public void TestUpdate_AvgPooling()
        {
            var layer = new PoolingLayer(3, 3, PoolingType.Average);
            layer.Build(NN.Size.Of(9));
            layer.OutputSize.Should().Be(NN.Size.Of(3));

            var result = layer.Update(new float[] { 1f, 2f, 3f, 5f, 1f, -3f, -1f, -2f, -3f });

            result.Should().HaveCount(3);
            result[0].Should().Be(2f);
            result[1].Should().Be(1f);
            result[2].Should().Be(-2f);
        }

        [TestCase(PoolingType.Max, 1, 2, 2)]
        [TestCase(PoolingType.Min, 1, -2, -2)]
        [TestCase(PoolingType.Sum, 1, 2, 3)]
        [TestCase(PoolingType.Average, 1, 2, 1.5f)]
        public void TestUpdate_LargeWindow(PoolingType type, float input1, float input2, float output)
        {
            var layer = new PoolingLayer(2, 1, type);
            layer.Build(NN.Size.Of(2));
            layer.OutputSize.Should().Be(NN.Size.Of(1));

            var result = layer.Update(new float[] { input1, input2 });

            result.Should().HaveCount(1);
            result[0].Should().Be(output);
        }

        [TestCase(PoolingType.Max, new[] { 1f, 2, 3, 4 }, new[] { 3f, 4 })]
        [TestCase(PoolingType.Min, new[] { 1f, 2, 3, 4 }, new[] { 1f, 2 })]
        [TestCase(PoolingType.Sum, new[] { 1f, 2, 3, 4 }, new[] { 6f, 9 })]
        [TestCase(PoolingType.Average, new[] { 1f, 2, 3, 4 }, new[] { 2f, 3 })]
        public void TestUpdate_DifferentStrideAndWindow(PoolingType type, float[] input, float[] output)
        {
            var layer = new PoolingLayer(3, 1, type);
            layer.Build(NN.Size.Of(input.Length));
            layer.OutputSize.Should().Be(NN.Size.Of(output.Length));

            var result = layer.Update(input);

            result.Should().HaveCount(output.Length);
            for (int i = 0; i < output.Length; i++)
            {
                result[i].Should().Be(output[i]);
            }
        }

        [TestCase(10, 3)]
        [TestCase(11, 3)]
        [TestCase(12, 4)]
        [TestCase(9, 3)]
        public void TestBuild(int inputSize, int expectedNodeCount)
        {
            var layer = new PoolingLayer(3, 3, PoolingType.Min);

            layer.Build(NN.Size.Of(inputSize));

            layer.Parameters.Get1dVectorNames().Should().BeEmpty();
            layer.Parameters.Get2dVectorNames().Should().BeEmpty();
            layer.State.Get1dVector("nodeValues").Should().HaveCount(expectedNodeCount);
            layer.OutputSize.Should().Be(NN.Size.Of(expectedNodeCount));
        }
    }
}
