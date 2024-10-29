using Ivankarez.NeuralNetworks.Abstractions;
using System;
using System.Collections.ObjectModel;

namespace Ivankarez.NeuralNetworks.Tensors
{
    public abstract class Tensor : ITensor
    {
        public int Rank { get; }
        public string Name { get; set; }
        public ReadOnlyCollection<int> Shape { get; }

        public Tensor(params int[] shape) : this(new ReadOnlyCollection<int>(shape))
        {
        }

        public Tensor(ReadOnlyCollection<int> shape)
        {
            Shape = shape ?? throw new ArgumentNullException(nameof(shape));
            Rank = shape.Count;
        }

        public override bool Equals(object obj)
        {
            if (obj is Tensor tensor)
            {
                if (Rank != tensor.Rank)
                {
                    return false;
                }

                for (int i = 0; i < Rank; i++)
                {
                    if (Shape[i] != tensor.Shape[i])
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
            return HashCode.Combine(Rank, Shape);
        }
    }
}
