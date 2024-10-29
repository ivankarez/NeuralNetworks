using System;

namespace Ivankarez.NeuralNetworks.Tensors
{
    public class Tensor2D : Tensor
    {
        private readonly Tensor1D[] values;

        public Tensor1D this[int index]
        {
            get => values[index];
            set => values[index] = value;
        }
        public float this[int i, int j]
        {
            get => values[i][j];
            set => values[i][j] = value;
        }
        public int Height { get; }
        public int Width { get; }

        public Tensor2D(Tensor1D[] values) : base(values.Length, values.Length == 0 ? 0 : values[0].Length)
        {
            this.values = values ?? throw new ArgumentNullException(nameof(values));
            Height = Shape[0];
            Width = Shape[1];
        }

        public override bool Equals(object obj)
        {
            if (obj is Tensor2D tensor)
            {
                if (tensor.Width != Width || tensor.Height != Height)
                {
                    return false;
                }

                for (int i = 0; i < Height; i++)
                {
                    if (!values[i].Equals(tensor.values[i]))
                    {
                        return false;
                    }
                }

                return true;
            }

            return false;
        }
    
        public override int GetHashCode()
        {
            return HashCode.Combine(values);
        }
    }
}
