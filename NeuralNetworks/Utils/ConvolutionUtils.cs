namespace Ivankarez.NeuralNetworks.Utils
{
    public static class ConvolutionUtils
    {
        public static int CalculateOutputSize(int inputSize, int window, int stride)
        {
            return (inputSize - window) / stride + 1;
        }
    }
}
