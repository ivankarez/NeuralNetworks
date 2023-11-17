using BenchmarkDotNet.Attributes;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Utils;

namespace Ivankarez.NeuralNetworks.Benchmarks.Layers
{
    [MemoryDiagnoser(false)]
    public class Pooling2dLayerBenchmark
    {
        private readonly Pooling2dLayer maxLayer;
        private readonly Pooling2dLayer minLayer;
        private readonly Pooling2dLayer avgLayer;
        private readonly Pooling2dLayer sumLayer;

        private readonly float[] inputs;

        public Pooling2dLayerBenchmark()
        {
            maxLayer = NN.Layers.Pooling2D(NN.Size.Of(3, 3));
            maxLayer.Build(NN.Size.Of(10, 10));

            minLayer = NN.Layers.Pooling2D(NN.Size.Of(3, 3), poolingType: PoolingType.Min);
            minLayer.Build(NN.Size.Of(10, 10));

            avgLayer = NN.Layers.Pooling2D(NN.Size.Of(3, 3), poolingType: PoolingType.Average);
            avgLayer.Build(NN.Size.Of(10, 10));

            sumLayer = NN.Layers.Pooling2D(NN.Size.Of(3, 3), poolingType: PoolingType.Sum);
            sumLayer.Build(NN.Size.Of(10, 10));

            inputs = NN.Random.Default().NextFloats(1f, 100).ToArray();
        }

        [Benchmark]
        public void Pooling2DMax() => maxLayer.Update(inputs);

        [Benchmark]
        public void Pooling2DMin() => minLayer.Update(inputs);

        [Benchmark]
        public void Pooling2DAverage() => avgLayer.Update(inputs);

        [Benchmark]
        public void Pooling2DSum() => sumLayer.Update(inputs);
    }
}
