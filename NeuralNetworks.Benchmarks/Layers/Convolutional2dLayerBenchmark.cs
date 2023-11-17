using BenchmarkDotNet.Attributes;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration;

namespace Ivankarez.NeuralNetworks.Benchmarks.Layers
{
    [MemoryDiagnoser(false)]
    public class Convolutional2dLayerBenchmark
    {
        private readonly Convolutional2dLayer layer;
        private readonly float[] input;

        public Convolutional2dLayerBenchmark()
        {
            layer = NN.Layers.Conv2D(NN.Size.Of(3, 3));
            layer.Build(NN.Size.Of(10, 10));

            input = NN.Random.Default().NextFloats(1f, 100);
        }

        [Benchmark]
        public void Convolutional2D() => layer.Update(input);
    }
}
