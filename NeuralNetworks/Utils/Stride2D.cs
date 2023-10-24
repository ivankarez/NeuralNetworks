using System;

namespace Ivankarez.NeuralNetworks.Utils
{
    public class Stride2D
    {
        public int Horizontal { get; }
        public int Vertical { get; }

        public Stride2D(int horizontal, int vertical)
        {
            if (horizontal <= 0) throw new ArgumentException("Horizontal stride must be a positive integer.", nameof(horizontal));
            if (vertical <= 0) throw new ArgumentException("Vertical stride must be a positive integer.", nameof(vertical));

            Horizontal = horizontal;
            Vertical = vertical;
        }

        public static implicit operator Stride2D((int, int) tuple)
        {
            return new Stride2D(tuple.Item1, tuple.Item2);
        }

        public static implicit operator Stride2D(int size)
        {
            return new Stride2D(size, size);
        }
    }
}
