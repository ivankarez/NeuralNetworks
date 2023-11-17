using BenchmarkDotNet.Attributes;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration;

namespace Ivankarez.NeuralNetworks.Benchmarks.Layers
{
    [MemoryDiagnoser(false)]
    public class DenseLayerBenchmark
    {
        private readonly DenseLayer layer;
        private readonly float[] input;

        public DenseLayerBenchmark()
        {
            layer = NN.Layers.Dense(100);
            layer.Build(NN.Size.Of(100));
            input = NN.Random.Default().NextFloats(1f, 100);
        }

        [Benchmark]
        public void Dense() => layer.Update(input);
    }
}
