# Benchmark Results
```
BenchmarkDotNet v0.13.10, Windows 11 (10.0.22635.2771)
AMD Ryzen 5 5600H with Radeon Graphics, 1 CPU, 12 logical and 6 physical cores
.NET SDK 7.0.304
  [Host]     : .NET 6.0.18 (6.0.1823.26907), X64 RyuJIT AVX2
  DefaultJob : .NET 6.0.18 (6.0.1823.26907), X64 RyuJIT AVX2


| Type                          | Method           | Mean       | Error     | StdDev   | Allocated |
|------------------------------ |----------------- |-----------:|----------:|---------:|----------:|
| Convolutional2dLayerBenchmark | Convolutional2D  | 1,286.7 ns |   4.46 ns |  4.17 ns |         - |
| ConvolutionalLayerBenchmark   | Convolutional    |   683.6 ns |   1.53 ns |  1.43 ns |         - |
| DenseLayerBenchmark           | Dense            | 9,292.3 ns | 101.87 ns | 95.29 ns |         - |
| Pooling2dLayerBenchmark       | Pooling2DMax     |   828.6 ns |   3.16 ns |  2.80 ns |         - |
| PoolingLayerBenchmark         | PoolingMax       |   756.8 ns |   4.43 ns |  4.15 ns |         - |
| RecurrentLayerBenchmark       | SimpleRecurrent  | 9,486.2 ns | 100.46 ns | 93.97 ns |         - |
| Pooling2dLayerBenchmark       | Pooling2DMin     |   787.9 ns |   2.40 ns |  2.25 ns |         - |
| PoolingLayerBenchmark         | PoolingMin       |   799.3 ns |   2.96 ns |  2.77 ns |         - |
| Pooling2dLayerBenchmark       | Pooling2DAverage |   756.6 ns |   1.89 ns |  1.68 ns |         - |
| PoolingLayerBenchmark         | PoolingAvg       |   609.3 ns |   3.25 ns |  2.88 ns |         - |
| Pooling2dLayerBenchmark       | Pooling2DSum     |   722.4 ns |   2.15 ns |  2.01 ns |         - |
| PoolingLayerBenchmark         | PoolingSum       |   487.8 ns |   4.87 ns |  4.55 ns |         - |
```
