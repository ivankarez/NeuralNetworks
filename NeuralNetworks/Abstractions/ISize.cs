namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface ISize
    {
        public int Dimensions { get; }
        public int this[int dimension] { get; }
        public int TotalSize { get; }
    }
}
