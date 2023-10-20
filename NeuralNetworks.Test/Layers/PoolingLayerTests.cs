using FluentAssertions;
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
            layer.Build(9);
            layer.NodeCount.Should().Be(3);

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
            layer.Build(9);
            layer.NodeCount.Should().Be(3);

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
            layer.Build(9);
            layer.NodeCount.Should().Be(3);

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
            layer.Build(9);
            layer.NodeCount.Should().Be(3);

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
            var layer = new PoolingLayer(3, 3, type);
            layer.Build(2);
            layer.NodeCount.Should().Be(1);

            var result = layer.Update(new float[] { input1, input2 });

            result.Should().HaveCount(1);
            result[0].Should().Be(output);
        }

        [TestCase(PoolingType.Max, new float[] { 1, 2, 3, 4 }, new float[] { 3, 4 })]
        [TestCase(PoolingType.Min, new float[] { 1, 2, 3, 4 }, new float[] { 1, 3 })]
        [TestCase(PoolingType.Sum, new float[] { 1, 2, 3, 4 }, new float[] { 6, 7 })]
        [TestCase(PoolingType.Average, new float[] { 1, 2, 3, 4 }, new float[] { 2, 3.5f })]
        public void TestUpdate_DifferentStrideAndWindow(PoolingType type, float[] input, float[] output)
        {
            var layer = new PoolingLayer(3, 2, type);
            layer.Build(input.Length);
            layer.NodeCount.Should().Be(output.Length);

            var result = layer.Update(input);

            result.Should().HaveCount(output.Length);
            for (int i = 0; i < output.Length; i++)
            {
                result[i].Should().Be(output[i]);
            }
        }

        [TestCase(10, 4)]
        [TestCase(11, 4)]
        [TestCase(12, 4)]
        [TestCase(9, 3)]
        [TestCase(1, 1)]
        public void TestBuild(int inputSize, int expectedNodeCount)
        {
            var layer = new PoolingLayer(3, 3, PoolingType.Min);

            layer.Build(inputSize);

            layer.Parameters.Get1dVectorNames().Should().BeEmpty();
            layer.Parameters.Get2dVectorNames().Should().BeEmpty();
            layer.State.Get1dVector("nodeValues").Should().HaveCount(expectedNodeCount);
            layer.NodeCount.Should().Be(expectedNodeCount);
        }
    }
}
