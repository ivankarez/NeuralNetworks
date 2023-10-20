namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IInitializer
    {
        public float GenerateValue(int fanIn, int fanOut);
    }
}
