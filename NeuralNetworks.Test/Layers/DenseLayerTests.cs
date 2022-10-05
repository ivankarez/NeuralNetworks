﻿using FluentAssertions;
using NeuralNetworks.Activations;
using NeuralNetworks.Layers;
using NUnit.Framework;

namespace NeuralNetworks.Test.Layers
{
    public class DenseLayerTests
    {
        [Test]
        public void TestUpdate_HappyPath()
        {
            var layer = new DenseLayer(2, new LinearActivation());
            var parameters = new ModelParameters(new float[] { -1f, 2.3f });
            layer.Build(1, parameters, new ModelParameters());

            var result = layer.Update(ParameterRange.Of(2f));

            result.Should().HaveCount(2);
            result[0].Should().Be(-2f);
            result[1].Should().Be(4.6f);
        }

        [Test]
        public void TestUpdate_HappyPath2()
        {
            var layer = new DenseLayer(3, new LinearActivation());
            var parameters = new ModelParameters(new float[] { -1f, 2.3f, 1.34f, .5f, -.34f, .2f });
            layer.Build(2, parameters, new ModelParameters());

            var result = layer.Update(ParameterRange.Of(2f, -.5f));

            result.Should().HaveCount(3);
            result[0].Should().BeApproximately(-3.15f, .01f);
            result[1].Should().BeApproximately(2.43f, .01f);
            result[2].Should().BeApproximately(-.78f, .01f);
        }
    }
}