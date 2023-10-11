using Ivankarez.NeuralNetworks.Values;

namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IModelLayer
    {
        public int NodeCount { get; }

        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        public void Build(int inputSize);

        public float[] Update(float[] inputValues);
    }
}
