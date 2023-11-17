using BenchmarkDotNet.Attributes;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration;

namespace Ivankarez.NeuralNetworks.Benchmarks.Layers
{
    [MemoryDiagnoser(false)]
    public class RecurrentLayerBenchmark
    {
        private readonly RecurrentLayer layer;
        private readonly float[] input;

        public RecurrentLayerBenchmark()
        {
            layer = NN.Layers.SimpleRecurrent(100);
            layer.Build(NN.Size.Of(100));
            input = NN.Random.Default().NextFloats(1f, 100);
        }

        [Benchmark]
        public void SimpleRecurrent() => layer.Update(input);
    }
}
