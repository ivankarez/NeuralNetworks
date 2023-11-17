using FluentAssertions;
using Ivankarez.NeuralNetworks.Activations;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;
using Ivankarez.NeuralNetworks.Test.Utils;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Layers
{
    public class GruLayerTests
    {
        [Test]
        public void TestConstructor_HappyPath()
        {
            var layer = NN.Layers.GRU(NN.Size.Of(3));

            layer.OutputSize.Should().Be(new Size1D(3));
            layer.Activation.Should().BeOfType<TanhActivation>();
            layer.RecurrentActivation.Should().BeOfType<SigmoidActivation>();
            layer.UseBias.Should().BeTrue();
            layer.KernelInitializer.Should().BeOfType<GlorotUniformInitializer>();
            layer.RecurrentInitializer.Should().BeOfType<GlorotUniformInitializer>();
            layer.BiasInitializer.Should().BeOfType<ZerosInitializer>();
        }

        [Test]
        public void TestConstructor_NullOutputSize()
        {
            Action call = () => NN.Layers.GRU(null);
            call.Should().Throw<ArgumentNullException>()
                .WithMessage("Value cannot be null. (Parameter 'nodeCount')");
        }

        [Test]
        public void TestBuild_HappyPath()
        {
            var layer = NN.Layers.GRU(NN.Size.Of(3),
                kernelInitializer: NN.Initializers.Constant(1),
                recurrentInitializer: NN.Initializers.Constant(2),
                biasInitializer: NN.Initializers.Constant(3));

            layer.Build(NN.Size.Of(2));

            layer.ForgetGateWeights.ShouldOnlyContain(1f);
            layer.CandidateWeights.ShouldOnlyContain(1f);
            layer.NodeValues.ShouldOnlyContain(0f);
            layer.ForgetRecurrentWeights.ShouldOnlyContain(2f);
            layer.CandidateRecurrentWeights.ShouldOnlyContain(2f);
            layer.ForgetBiases.ShouldOnlyContain(3f);
            layer.CandidateBiases.ShouldOnlyContain(3f);

            layer.Parameters.ShouldContain2D("forgetGateWeights").Should().BeEquivalentTo(layer.ForgetGateWeights);
            layer.Parameters.ShouldContain2D("candidateWeights").Should().BeEquivalentTo(layer.CandidateWeights);
            layer.Parameters.ShouldContain1D("forgetRecurrentWeights").Should().BeEquivalentTo(layer.ForgetRecurrentWeights);
            layer.Parameters.ShouldContain1D("candidateRecurrentWeights").Should().BeEquivalentTo(layer.CandidateRecurrentWeights);
            layer.Parameters.ShouldContain1D("forgetBiases").Should().BeEquivalentTo(layer.ForgetBiases);
            layer.Parameters.ShouldContain1D("candidateBiases").Should().BeEquivalentTo(layer.CandidateBiases);
            layer.State.ShouldContain1D("nodeValues").Should().BeEquivalentTo(layer.NodeValues);
        }

        [Test]
        public void TestBuild_WithoutBias()
        {
            var layer = NN.Layers.GRU(NN.Size.Of(3),
                kernelInitializer: NN.Initializers.Constant(1),
                recurrentInitializer: NN.Initializers.Constant(2),
                useBias: false);

            layer.Build(NN.Size.Of(2));

            layer.ForgetGateWeights.ShouldOnlyContain(1f);
            layer.CandidateWeights.ShouldOnlyContain(1f);
            layer.NodeValues.ShouldOnlyContain(0f);
            layer.ForgetRecurrentWeights.ShouldOnlyContain(2f);
            layer.CandidateRecurrentWeights.ShouldOnlyContain(2f);
            layer.ForgetBiases.Should().BeNull();
            layer.CandidateBiases.Should().BeNull();

            layer.Parameters.ShouldContain2D("forgetGateWeights").Should().BeEquivalentTo(layer.ForgetGateWeights);
            layer.Parameters.ShouldContain2D("candidateWeights").Should().BeEquivalentTo(layer.CandidateWeights);
            layer.Parameters.ShouldContain1D("forgetRecurrentWeights").Should().BeEquivalentTo(layer.ForgetRecurrentWeights);
            layer.Parameters.ShouldContain1D("candidateRecurrentWeights").Should().BeEquivalentTo(layer.CandidateRecurrentWeights);
            layer.Parameters.ShouldNotContain1D("forgetBiases");
            layer.Parameters.ShouldNotContain1D("candidateBiases");
            layer.State.ShouldContain1D("nodeValues").Should().BeEquivalentTo(layer.NodeValues);
        }

        [Test]
        public void TestBuild_NullInputSize()
        {
            var layer = NN.Layers.GRU(NN.Size.Of(3));
            Action call = () => layer.Build(null);
            call.Should().Throw<ArgumentNullException>()
                .WithMessage("Value cannot be null. (Parameter 'inputSize')");
        }

        [Test]
        public void TestUpdate_TrivialCase()
        {
            var layer = NN.Layers.GRU(NN.Size.Of(1),
                kernelInitializer: NN.Initializers.Constant(1),
                recurrentInitializer: NN.Initializers.Constant(0),
                useBias: false);

            layer.Build(NN.Size.Of(1));

            var output = layer.Update(new[] { 1f });

            output.Should().BeEquivalentTo(new[] { 0.55676997f });
        }

        [Test]
        public void TestUpdate_UsesBias()
        {
            var layer = NN.Layers.GRU(NN.Size.Of(1),
                kernelInitializer: NN.Initializers.Constant(1),
                recurrentInitializer: NN.Initializers.Constant(0),
                biasInitializer: NN.Initializers.Constant(1));

            layer.Build(NN.Size.Of(1));

            var output = layer.Update(new[] { 1f });

            output.Should().BeEquivalentTo(new[] { 0.84911263f });
        }

        [Test]
        public void TestUpdate_UsesRecurrent()
        {
            var layer = NN.Layers.GRU(NN.Size.Of(1),
                kernelInitializer: NN.Initializers.Constant(1),
                recurrentInitializer: NN.Initializers.Constant(1),
                useBias: false);

            layer.Build(NN.Size.Of(1));
            layer.NodeValues[0] = 1f;

            var output = layer.Update(new[] { 1f });

            output.Should().BeEquivalentTo(new[] { 0.9599792f });
        }

        [Test]
        public void TestUpdate_NotIdempotent()
        {
            var layer = NN.Layers.GRU(NN.Size.Of(1),
                kernelInitializer: NN.Initializers.Constant(1),
                recurrentInitializer: NN.Initializers.Constant(1),
                useBias: false);

            layer.Build(NN.Size.Of(1));
            layer.NodeValues[0] = 1f;

            var input = new[] { 1f };
            var output1 = layer.Update(input).Clone();
            var output2 = layer.Update(input);

            output1.Should().NotBeEquivalentTo(output2);
        }

        [Test]
        public void TestUpdate_BigNetwork()
        {
            var randomProvider = NN.Random.System(new Random(0));
            var initializer = NN.Initializers.GlorotUniform(randomProvider);
            var layer = NN.Layers.GRU(NN.Size.Of(10),
                kernelInitializer: initializer,
                recurrentInitializer: initializer,
                biasInitializer: initializer);

            layer.Build(NN.Size.Of(100));

            var input = randomProvider.NextFloats(1f, 100);
            var output = layer.Update(input);

            var expectedOutput = new[] { 0.35179606f, 0.03638931f, -0.34586656f, -0.4565702f,
                -0.22304979f, 0.17532094f, 0.14402571f, -0.04157398f, 0.068718806f, -0.027657902f };
            output.Should().BeEquivalentTo(expectedOutput);
        }
    }
}
