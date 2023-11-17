# Benchmark Results
```
BenchmarkDotNet v0.13.10, Windows 11 (10.0.22635.2771)
AMD Ryzen 5 5600H with Radeon Graphics, 1 CPU, 12 logical and 6 physical cores
.NET SDK 7.0.304
  [Host]     : .NET 6.0.18 (6.0.1823.26907), X64 RyuJIT AVX2
  DefaultJob : .NET 6.0.18 (6.0.1823.26907), X64 RyuJIT AVX2


| Type                          | Method           | Mean        | Error     | StdDev    | Allocated |
|------------------------------ |----------------- |------------:|----------:|----------:|----------:|
| Convolutional2dLayerBenchmark | Convolutional2D  |  1,229.1 ns |   5.54 ns |   5.19 ns |         - |
| ConvolutionalLayerBenchmark   | Convolutional    |    686.0 ns |   2.88 ns |   2.69 ns |         - |
| DenseLayerBenchmark           | Dense            |  9,336.5 ns |  90.28 ns |  84.45 ns |         - |
| GruLayerBenchmark             | GRU              | 19,146.6 ns | 132.64 ns | 117.58 ns |         - |
| Pooling2dLayerBenchmark       | Pooling2DMax     |    832.4 ns |   4.00 ns |   3.74 ns |         - |
| PoolingLayerBenchmark         | PoolingMax       |    761.1 ns |   4.94 ns |   4.62 ns |         - |
| RecurrentLayerBenchmark       | SimpleRecurrent  |  9,368.3 ns |  63.06 ns |  55.90 ns |         - |
| Pooling2dLayerBenchmark       | Pooling2DMin     |    828.6 ns |   2.34 ns |   1.96 ns |         - |
| PoolingLayerBenchmark         | PoolingMin       |    846.6 ns |   2.13 ns |   1.99 ns |         - |
| Pooling2dLayerBenchmark       | Pooling2DAverage |    764.6 ns |   2.96 ns |   2.77 ns |         - |
| PoolingLayerBenchmark         | PoolingAvg       |    614.6 ns |   9.51 ns |   8.90 ns |         - |
| Pooling2dLayerBenchmark       | Pooling2DSum     |    724.4 ns |   2.56 ns |   2.40 ns |         - |
| PoolingLayerBenchmark         | PoolingSum       |    484.9 ns |   1.72 ns |   1.44 ns |         - |
```
