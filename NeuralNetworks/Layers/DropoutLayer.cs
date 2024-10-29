using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DropoutLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; }
        public float DropoutRate { get; }
        public IRandomProvider RandomProvider { get; }
        public float[] Output { get; private set; }

        public DropoutLayer(float dropoutRate, IRandomProvider randomProvider)
        {
            DropoutRate = dropoutRate;
            RandomProvider = randomProvider;
        }

        public void Build(ISize inputSize)
        {
            OutputSize = inputSize;
            Output = new float[OutputSize.TotalSize];
        }

        public float[] Update(float[] inputValues)
        {
            for (int i = 0; i < OutputSize.TotalSize; i++)
            {
                Output[i] =  RandomProvider.NextBool(DropoutRate) ? inputValues[i] : 0;
            }

            return Output;
        }
    }
}
