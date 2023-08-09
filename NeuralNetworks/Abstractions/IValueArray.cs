namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IValueArray : IReadonlyValueArray
    {
        public new float this[int index]
        {
            get;
            set;
        }
    }
}
