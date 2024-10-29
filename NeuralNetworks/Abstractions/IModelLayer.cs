namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IModelLayer
    {
        public ISize OutputSize { get; }

        public void Build(ISize inputSize);

        public float[] Update(float[] inputValues);

        // TODO: Add paramter count property
    }
}
