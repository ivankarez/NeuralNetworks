using System;
using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Utils
{
    public class Size2D : ISize
    {
        public int Dimensions => 2;
        public int TotalSize { get; }
        public int Width { get; }
        public int Height { get; }

        public int this[int dimension]
        {
            get
            {
                if (dimension == 0) return Width;
                if (dimension == 1) return Height;
                throw new IndexOutOfRangeException($"Dimension {dimension} is not valid on Size2D");
            }
        }

        public Size2D(int width, int height)
        {
            if (width <= 0) throw new ArgumentException("Width must be a positive integer.", nameof(width));
            if (height <= 0) throw new ArgumentException("Height must be a positive integer.", nameof(height));

            Width = width;
            Height = height;
            TotalSize = width * height;
        }

        public static implicit operator Size2D((int, int) tuple)
        {
            return new Size2D(tuple.Item1, tuple.Item2);
        }

        public static implicit operator Size2D(int size)
        {
            return new Size2D(size, size);
        }

        public override bool Equals(object obj)
        {
            return obj is Size2D size && Equals(size);
        }

        public bool Equals(Size2D other)
        {
            return Width == other.Width &&
                   Height == other.Height;
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(Width, Height);
        }
    }
}
