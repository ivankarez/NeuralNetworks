using Ivankarez.NeuralNetworks.Abstractions;
using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks.Extensions
{
    public static class IValueArrayExtensions
    {
        public static void SetValues(this IValueArray array, IEnumerable<float> values)
        {
            var index = 0;
            foreach (var value in values)
            {
                array[index++] = value;
            }
        }
    }
}
