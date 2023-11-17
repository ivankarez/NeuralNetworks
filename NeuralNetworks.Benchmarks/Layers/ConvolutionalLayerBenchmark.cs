using BenchmarkDotNet.Attributes;
using Ivankarez.NeuralNetworks.Api;
using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.RandomGeneration;

namespace Ivankarez.NeuralNetworks.Benchmarks.Layers
{
    [MemoryDiagnoser(false)]
    public class ConvolutionalLayerBenchmark
    {
        private readonly ConvolutionalLayer layer;
        private readonly float[] input;

        public ConvolutionalLayerBenchmark()
        {
            layer = NN.Layers.Conv1D(10);
            layer.Build(NN.Size.Of(100));

            input = NN.Random.Default().NextFloats(1f, 100);
        }

        [Benchmark]
        public void Convolutional() => layer.Update(input);
    }
}
