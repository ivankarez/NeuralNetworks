using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
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
            var layer = NN.Layers.Conv1D(3, useBias: false);

            layer.Build(new Size1D(3));

            layer.Parameters.Get1dVector("filter").Should().HaveCount(3);
            layer.State.Get1dVector("nodeValues").Should().HaveCount(1);
        }

        [Test]
        public void TestBuild_UseInitializer()
        {
            var layer = NN.Layers.Conv1D(3, kernelInitializer: new ConstantInitializer(3f), biasInitializer: new ConstantInitializer(2f));
            layer.Build(new Size1D(3));
            layer.Parameters.Get1dVector("filter").Should().AllBeEquivalentTo(3);
            layer.Parameters.Get1dVector("biases").Should().AllBeEquivalentTo(2);
        }

        [Test]
        public void TestBuild_TooBigFilter()
        {
            var layer = NN.Layers.Conv1D(4);

            Action callAction = () => layer.Build(new Size1D(3));

            callAction.Should().Throw<ArgumentException>();
        }

        [Test]
        public void TestBuild_TooSmallFilter()
        {
            Action callAction = () => NN.Layers.Conv1D(0);
            callAction.Should().Throw<ArgumentException>();
        }

        [Test]
        public void TestUpdate_SingleKernel()
        {
            var layer = NN.Layers.Conv1D(3, useBias: false);
            layer.Build(new Size1D(3));
            layer.Parameters.Get1dVector("filter").Fill(1, 2, -1);

            var result = layer.Update(new float[] { 3, -5, -4 });

            result.Should().HaveCount(1);
            result[0].Should().Be(-3);
        }

        [Test]
        public void TestUpdate_MultipleKernel()
        {
            var layer = NN.Layers.Conv1D(2, useBias: false);
            layer.Build(new Size1D(6));
            layer.Parameters.Get1dVector("filter").Fill(1, -2);

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
            var layer = NN.Layers.Conv1D(1, useBias: false);
            layer.Build(new Size1D(3));
            layer.Parameters.Get1dVector("filter").Fill(.5f);

            var result = layer.Update(new float[] { 3, -5, 0 });

            result.Should().HaveCount(3);
            result[0].Should().Be(1.5f);
            result[1].Should().Be(-2.5f);
            result[2].Should().Be(0f);
        }

        [Test]
        public void TestUpdate_UseBias()
        {
            var layer = NN.Layers.Conv1D(2, kernelInitializer: new ConstantInitializer(1), biasInitializer: new ConstantInitializer(2));
            layer.Build(new Size1D(4));

            var result = layer.Update(new float[] { 3, -5, -4, 2 });
            result.Should().HaveCount(3);
            result.Should().BeEquivalentTo(new[] { 0f, -7f, 0f });
        }

        [Test]
        public void TestUpdate_WithStride()
        {
            var layer = NN.Layers.Conv1D(2, stride: 2, useBias: false);
            layer.Build(new Size1D(6));
            layer.Parameters.Get1dVector("filter").Fill(1, -2);
            var result = layer.Update(new[] { 3f, -5f, -4f, 2f, 3f, 1f });
            result.Should().BeEquivalentTo(new[] { 13f, -8f, 1f });
        }
    }
}
