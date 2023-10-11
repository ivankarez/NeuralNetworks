using System;

namespace Ivankarez.NeuralNetworks.Utils
{
    public static class ArrayExtensions
    {
        public static T[] Fill<T>(this T[] array, params T[] values)
        {
            if (array.Length != values.Length)
            {
                throw new ArgumentException("Array size does not match");
            }

            for (int i = 0; i < array.Length; i++)
            {
                array[i] = values[i];
            }
            return array;
        }

        public static T[,] Fill<T>(this T[,] array, T[,] values)
        {
            if (array.Length != values.Length)
            {
                throw new ArgumentException("Array size does not match");
            }

            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int j = 0; j < array.GetLength(1); j++)
                {
                    array[i, j] = values[i, j];
                }
            }
            return array;
        }
    }
}
