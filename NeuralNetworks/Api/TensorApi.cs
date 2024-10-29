using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Tensors;

namespace Ivankarez.NeuralNetworks.Api
{
    public class TensorApi
    {
        internal TensorApi() { }

        public Tensor1D Of(float[] values)
        {
            return new Tensor1D(values);
        }
        public Tensor1D Zeros(int size)
        {
            var values = new float[size];
            return new Tensor1D(values);
        }
        public Tensor1D Random(int size, float min = 0f, float max = 1f, IRandomProvider randomProvider = null)
        {
            randomProvider ??= NN.Random.Default();
            var values = randomProvider.NextFloats(min, max, size);
            return new Tensor1D(values);
        }

        public Tensor2D Of(float[,] values)
        {
            var rows = new Tensor1D[values.GetLength(0)];
            for (int i = 0; i < values.GetLength(0); i++)
            {
                var row = new float[values.GetLength(1)];
                for (int j = 0; j < values.GetLength(1); j++)
                {
                    row[j] = values[i, j];
                }
                rows[i] = new Tensor1D(row);
            }

            return new Tensor2D(rows);
        }
        public Tensor2D Zeros(int rows, int columns)
        {
            var values = new float[rows, columns];
            return Of(values);
        }
        public Tensor2D Random(int rows, int columns, float min = 0f, float max = 1f, IRandomProvider randomProvider = null)
        {
            randomProvider ??= NN.Random.Default();
            var values = new Tensor1D[rows];
            for (int i = 0; i < rows; i++)
            {
                values[i] = Random(columns, min, max, randomProvider);
            }

            return new Tensor2D(values);
        }
    }
}
