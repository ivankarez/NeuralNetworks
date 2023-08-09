using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Values;

namespace Ivankarez.NeuralNetworks.Utils
{
    public static class MathUtils
    {
        public static void ElementwiseMultiply(IValueArray a, IValueArray b, float[] result)
        {
            for (int i = 0; i < a.Count; i++)
            {
                result[i] = a[i] * b[i];
            }
        }

        public static void ElementwiseMultiplyAdditive(IValueArray a, IValueArray b, float[] result)
        {
            for (int i = 0; i < a.Count; i++)
            {
                result[i] += a[i] * b[i];
            }
        }

        internal static void ElementwiseMultiply(ValueStoreRange a, float b, ValueStoreRange result)
        {
            for (int i = 0; i < a.Count; i++)
            {
                result[i] = a[i] * b;
            }
        }
    }
}
