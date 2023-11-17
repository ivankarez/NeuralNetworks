using BenchmarkDotNet.Attributes;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration;

namespace Ivankarez.NeuralNetworks.Benchmarks.Layers
{
    [MemoryDiagnoser(false)]
    public class GruLayerBenchmark
    {
        private readonly GruLayer layer;
        private readonly float[] input;

        public GruLayerBenchmark()
        {
            layer = NN.Layers.GRU(NN.Size.Of(100));
            layer.Build(NN.Size.Of(100));
            input = NN.Random.Default().NextFloats(1f, 100);
        }

        [Benchmark]
        public void GRU() => layer.Update(input);
    }
}
