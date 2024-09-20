using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration;
using Ivankarez.NeuralNetworks.Values;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class DropoutLayer : IModelLayer
    {
        public ISize OutputSize { get; private set; }
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }
        public float DropoutRate { get; }
        public IRandomProvider RandomProvider { get; }

        private float[] nodeValues;

        public DropoutLayer(float dropoutRate, IRandomProvider randomProvider)
        {
            DropoutRate = dropoutRate;
            RandomProvider = randomProvider;
            Parameters = new NamedVectors<float>();
            State = new NamedVectors<float>();
        }

        public void Build(ISize inputSize)
        {
            OutputSize = inputSize;
            nodeValues = new float[OutputSize.TotalSize];
            State.Add("nodeValues", nodeValues);
        }

        public float[] Update(float[] inputValues)
        {
            for (int i = 0; i < OutputSize.TotalSize; i++)
            {
                nodeValues[i] =  RandomProvider.NextBool(DropoutRate) ? inputValues[i] : 0;
            }

            return nodeValues;
        }
    }
}
