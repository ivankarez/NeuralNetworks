using Ivankarez.NeuralNetworks.Values;

namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IModelLayer
    {
        public int NodeCount { get; }

        public void Build(int prevLayerSize, ValueStore parameters, ValueStore state);

        public IValueArray Update(IValueArray inputValues);
    }
}
