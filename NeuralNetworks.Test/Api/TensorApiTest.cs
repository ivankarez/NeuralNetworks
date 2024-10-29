using System;
using FluentAssertions;
using Ivankarez.NeuralNetworks.Api;
using NUnit.Framework;

namespace Ivankarez.NeuralNetworks.Test.Api
{
    public class TensorApiTest
    {
        private readonly TensorApi api;

        public TensorApiTest()
        {
            api = NN.Tensor;
        }

        [Test]
        public void TestOf_1D_Empty()
        {
            var tensor = api.Of(Array.Empty<float>());

            tensor.Length.Should().Be(0);
            tensor.Rank.Should().Be(1);
            tensor.Shape.Should().BeEquivalentTo(new[] { 0 });
        }

        [Test]
        public void TestOf_1D()
        {
            var tensor = api.Of(new[] { 1f, 2f, 3f });

            tensor.Length.Should().Be(3);
            tensor.Rank.Should().Be(1);
            tensor.Shape.Should().BeEquivalentTo(new[] { 3 });
            tensor[0].Should().Be(1f);
            tensor[1].Should().Be(2f);
            tensor[2].Should().Be(3f);
        }

        [Test]
        public void TestOf_2D_Empty()
        {
            var tensor = api.Of(new float[0, 0]);

            tensor.Width.Should().Be(0);
            tensor.Height.Should().Be(0);
            tensor.Rank.Should().Be(2);
            tensor.Shape.Should().BeEquivalentTo(new[] { 0, 0 });
        }

        [Test]
        public void TestOf_2D()
        {
            var tensor = api.Of(new float[2, 3]
            {
                { 1f, 2f, 3f },
                { 4f, 5f, 6f }
            });

            tensor.Width.Should().Be(3);
            tensor.Height.Should().Be(2);
            tensor.Rank.Should().Be(2);
            tensor.Shape.Should().BeEquivalentTo(new[] { 2, 3 });
            tensor[0, 0].Should().Be(1f);
            tensor[0, 1].Should().Be(2f);
            tensor[0, 2].Should().Be(3f);
            tensor[1, 0].Should().Be(4f);
            tensor[1, 1].Should().Be(5f);
            tensor[1, 2].Should().Be(6f);
        }

        [Test]
        public void TestZeros_1D()
        {
            var tensor = api.Zeros(3);

            tensor.Length.Should().Be(3);
            tensor.Rank.Should().Be(1);
            tensor.Shape.Should().BeEquivalentTo(new[] { 3 });
            tensor[0].Should().Be(0f);
            tensor[1].Should().Be(0f);
            tensor[2].Should().Be(0f);
        }

        [Test]
        public void TestZeros_2D()
        {
            var tensor = api.Zeros(2, 3);

            tensor.Width.Should().Be(3);
            tensor.Height.Should().Be(2);
            tensor.Rank.Should().Be(2);
            tensor.Shape.Should().BeEquivalentTo(new[] { 2, 3 });
            tensor[0, 0].Should().Be(0f);
            tensor[0, 1].Should().Be(0f);
            tensor[0, 2].Should().Be(0f);
            tensor[1, 0].Should().Be(0f);
            tensor[1, 1].Should().Be(0f);
            tensor[1, 2].Should().Be(0f);
        }

        [Test]
        public void TestRandom_1D()
        {
            var randomProvider = NN.Random.System(new Random(42));
            var tensor = api.Random(3, randomProvider: randomProvider);

            tensor.Length.Should().Be(3);
            tensor.Rank.Should().Be(1);
            tensor.Shape.Should().BeEquivalentTo(new[] { 3 });
            tensor[0].Should().Be(0.66810644f);
            tensor[1].Should().Be(0.1409073f);
            tensor[2].Should().Be(0.12551829f);
        }

        [Test]
        public void TestRandom_2D()
        {
            var randomProvider = NN.Random.System(new Random(42));
            var tensor = api.Random(2, 3, randomProvider: randomProvider);

            tensor.Width.Should().Be(3);
            tensor.Height.Should().Be(2);
            tensor.Rank.Should().Be(2);
            tensor.Shape.Should().BeEquivalentTo(new[] { 2, 3 });
            tensor[0, 0].Should().Be(0.66810644f);
            tensor[0, 1].Should().Be(0.1409073f);
            tensor[0, 2].Should().Be(0.12551829f);
            tensor[1, 0].Should().Be(0.522764266f);
            tensor[1, 1].Should().Be(0.168434218f);
            tensor[1, 2].Should().Be(0.262592673f);
        }
    }
}
