using Ivankarez.NeuralNetworks.Abstractions;

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
    }
}
