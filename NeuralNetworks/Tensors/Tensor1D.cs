using System;

namespace Ivankarez.NeuralNetworks.Tensors
{
    public class Tensor1D : Tensor
    {
        private readonly float[] values;

        public float this[int index]
        {
            get => values[index];
            set => values[index] = value;
        }
        public int Length { get; }

        public Tensor1D(float[] values) : base(values.Length)
        {
            this.values = values ?? throw new ArgumentNullException(nameof(values));
            Length = values.Length;
        }

        public override bool Equals(object obj)
        {
            if (obj is Tensor1D tensor)
            {
                if (tensor.values.Length != values.Length)
                {
                    return false;
                }

                for (int i = 0; i < values.Length; i++)
                {
                    if (values[i] != tensor.values[i])
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
