using FluentAssertions;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class ConvolutionalLayerTests
    {
        [Test]
        public void TestBuild_HappyPath()
        {
            var layer = new ConvolutionalLayer(3);

            layer.Build(3);

            layer.Parameters.Get1dVector("filter").Should().HaveCount(3);
            layer.State.Get1dVector("nodeValues").Should().HaveCount(1);
        }

        [Test]
        public void TestBuild_TooBigFilter()
        {
            var layer = new ConvolutionalLayer(4);

            Action callAction = () => layer.Build(3);

            callAction.Should().Throw<ArgumentException>();
        }

        [Test]
        public void TestBuild_TooSmallFilter()
        {
            Action callAction = () => new ConvolutionalLayer(0);
            callAction.Should().Throw<ArgumentException>();
        }

        [Test]
        public void TestUpdate_SingleKernel()
        {
            var layer = new ConvolutionalLayer(3);
            layer.Build(3);
            layer.Parameters.Get1dVector("filter").Fill(1, 2, -1 );

            var result = layer.Update(new float[] { 3, -5, -4 });

            result.Should().HaveCount(1);
            result[0].Should().Be(-3);
        }

        [Test]
        public void TestUpdate_MultipleKernel()
        {
            var layer = new ConvolutionalLayer(2);
            layer.Build(6);
            layer.Parameters.Get1dVector("filter").Fill(1, -2 );

            var result = layer.Update(new float[] { 3, -5, -4, 2, 3, 1 });

            result.Should().HaveCount(5);
            result[0].Should().Be(13);
            result[1].Should().Be(3);
            result[2].Should().Be(-8);
            result[3].Should().Be(-4);
            result[4].Should().Be(1);
        }

        [Test]
        public void TestUpdate_ShortestFilter()
        {
            var layer = new ConvolutionalLayer(1);
            layer.Build(3);
            layer.Parameters.Get1dVector("filter").Fill(.5f );

            var result = layer.Update(new float[] { 3, -5, 0 });

            result.Should().HaveCount(3);
            result[0].Should().Be(1.5f);
            result[1].Should().Be(-2.5f);
            result[2].Should().Be(0f);
        }
    }
}
