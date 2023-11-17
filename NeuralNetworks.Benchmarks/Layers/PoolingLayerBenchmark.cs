using BenchmarkDotNet.Attributes;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;

namespace Ivankarez.NeuralNetworks.Benchmarks.Layers
{
    [MemoryDiagnoser(false)]
    public class PoolingLayerBenchmark
    {
        private readonly PoolingLayer maxLayer;
        private readonly PoolingLayer minLayer;
        private readonly PoolingLayer avgLayer;
        private readonly PoolingLayer sumLayer;
        private readonly float[] input;

        public PoolingLayerBenchmark()
        {
            maxLayer = NN.Layers.Pooling1D(10);
            maxLayer.Build(NN.Size.Of(100));

            minLayer = NN.Layers.Pooling1D(10, poolingType: PoolingType.Min);
            minLayer.Build(NN.Size.Of(100));

            avgLayer = NN.Layers.Pooling1D(10, poolingType: PoolingType.Average);
            avgLayer.Build(NN.Size.Of(100));

            sumLayer = NN.Layers.Pooling1D(10, poolingType: PoolingType.Sum);
            sumLayer.Build(NN.Size.Of(100));

            input = NN.Random.Default().NextFloats(1f, 100);
        }

        [Benchmark]
        public void PoolingMax() => maxLayer.Update(input);

        [Benchmark]
        public void PoolingMin() => minLayer.Update(input);

        [Benchmark]
        public void PoolingAvg() => avgLayer.Update(input);

        [Benchmark]
        public void PoolingSum() => sumLayer.Update(input);
    }
}
