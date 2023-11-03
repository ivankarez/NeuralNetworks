using System;
using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Utils
{
    public class Size1D : ISize
    {
        public int this[int dimension]
        {
            get
            {
                if (dimension == 0) return TotalSize;
                throw new IndexOutOfRangeException($"Dimension {dimension} is not valid on Size1D");
            }
        }

        public int Dimensions => 1;

        public int TotalSize { get; }

        public Size1D(int size)
        {
            if (size <= 0) throw new ArgumentException("Size must be a positive integer.", nameof(size));
            TotalSize = size;
        }

        public override bool Equals(object obj)
        {
            return obj is Size1D size && TotalSize == size.TotalSize;
        }

        public override int GetHashCode()
        {
            return TotalSize.GetHashCode();
        }
    }
}
