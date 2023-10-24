using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class Pooling2dLayerTests
    {
        [Test]
        public void Build_HappyPath()
        {
            var layer = new Pooling2dLayer((3, 3), (2, 2), (1, 1), PoolingType.Max);
            layer.Build(9);
            layer.State.Get1dVector("nodeValues").Should().HaveCount(4);
            layer.Parameters.Get1dVectorNames().Should().BeEmpty();
            layer.Parameters.Get2dVectorNames().Should().BeEmpty();
        }

        [Test]
        public void Build_InvalidInputSize()
        {
            var layer = new Pooling2dLayer((3, 3), (2, 2), (1, 1), PoolingType.Max);
            layer.Invoking(l => l.Build(10)).Should().Throw<ArgumentException>();
        }

        [TestCase(PoolingType.Min, new float[] { -2, -3, 0, -9 })]
        [TestCase(PoolingType.Max, new float[] { 2, 3, 9, 9 })]
        [TestCase(PoolingType.Average, new float[] { 0, 0, 3, 1.25f })]
        [TestCase(PoolingType.Sum, new float[] { 0, 0, 12, 5 })]
        public void Update_HappyPath(PoolingType poolingType, float[] expectedResults)
        {
            var layer = new Pooling2dLayer((3, 3), (2, 2), (1, 1), poolingType);
            layer.Build(9);
            var inputs = new float[] {
                -1, -2, -3,
                1, 2, 3,
                0, 9, -9 
            };
            var result = layer.Update(inputs);
            result.Should().BeEquivalentTo(expectedResults);
        }

        [Test]
        public void Update_CheckStride()
        {
            var layer = NN.Layers.Pooling2D((5, 6), (2, 2), (2, 3));
            layer.Build(30);

            var inputs = new float[] { 
                0, 1, 0, 1, 2,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 3, 0, 1, 4,
                0, 6, 0, 0, 7,
                0, 0, 0, 0, 0,
            };
            var result = layer.Update(inputs);

            result.Should().BeEquivalentTo(new[] { 1f, 2f, 3f, 4f });
        }
    }
}
