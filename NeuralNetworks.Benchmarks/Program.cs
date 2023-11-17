using BenchmarkDotNet.Running;

namespace Ivankarez.NeuralNetworks.Benchmarks
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var switcher = BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly);
            // var switcher = BenchmarkSwitcher.FromTypes(new[] { typeof(Convolutional2dLayerBenchmark) });
            var result = switcher.RunAllJoined();
            Console.WriteLine(result);
        }
    }
}