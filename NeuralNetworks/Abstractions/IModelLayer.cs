using Ivankarez.NeuralNetworks.Values;

namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IModelLayer
    {
        public ISize OutputSize { get; }

        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        public void Build(ISize inputSize);

        public float[] Update(float[] inputValues);
    }
}
