namespace Ivankarez.NeuralNetworks.Api
{
    public static class NN
    {
        public static ModelsApi Models { get; } = new ModelsApi();
        public static LayersApi Layers { get; } = new LayersApi();
        public static ActivationsApi Activations { get; } = new ActivationsApi();
        public static RandomApi Random { get; } = new RandomApi();
        public static InitializersApi Initializers { get; } = new InitializersApi();
    }
}
