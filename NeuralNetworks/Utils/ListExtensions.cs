using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks.Utils
{
    public static class ListExtensions
    {
        public static void AddRange<T>(this List<T> list, T[,] items)
        {
            foreach (var item in items)
            {
                list.Add(item);
            }
        }
    }
}
