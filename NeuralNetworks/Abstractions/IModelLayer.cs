using Ivankarez.NeuralNetworks.Values;

namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IModelLayer
    {
        public ISize OutputSize { get; }

        // TODO: Remove these
        public NamedVectors<float> Parameters { get; }
        public NamedVectors<float> State { get; }

        public void Build(ISize inputSize);

        public float[] Update(float[] inputValues);

        // TODO: Add paramter count property
    }
}
