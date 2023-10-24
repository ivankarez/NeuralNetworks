using System;

namespace Ivankarez.NeuralNetworks.Utils
{
    public class Size2D
    {
        public int Width { get; }
        public int Height { get; }

        public Size2D(int width, int height)
        {
            if (width <= 0) throw new ArgumentException("Width must be a positive integer.", nameof(width));
            if (height <= 0) throw new ArgumentException("Height must be a positive integer.", nameof(height));

            Width = width;
            Height = height;
        }

        public static implicit operator Size2D((int, int) tuple)
        {
            return new Size2D(tuple.Item1, tuple.Item2);
        }

        public static implicit operator Size2D(int size)
        {
            return new Size2D(size, size);
        }
    }
}
