﻿using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Activations
{
    public class LeakyReluActivationTests
    {
        [TestCase(new float[] { 0f }, 0f)]
        [TestCase(new float[] { .5f }, .5f)]
        [TestCase(new float[] { 1f }, 1f)]
        [TestCase(new float[] { 10f }, 10f)]
        [TestCase(new float[] { -0.01f }, -.01f * .1f)]
        [TestCase(new float[] { -10f }, -10f * .1f)]
        [TestCase(new float[] { 2f, 3f, -10f }, 4f)]
        public void TestApply(float[] inputs, float expectedOutput)
        {
            var output = new LeakyReluActivation(.1f).Apply(inputs);
            output.Should().Be(expectedOutput);
        }
    }
}
