namespace NeuralNetworks.Abstractions
{
    public interface IModelLayer
    {
        public int NodeCount { get; }

        public void Build(int prevLayerSize, ModelParameters parameterBuilder, ModelParameters valueBuilder);

        public ParameterRange Update(ParameterRange inputValues);
    }
}
