namespace Ivankarez.NeuralNetworks.Api
{
    /// <summary>
    /// Provides access to the NeuralNetworks API.
    /// </summary>
    public static class NN
    {
        /// <summary>
        /// Gets an instance of the ModelsApi class, which provides access to a collection of pre-defined neural network models.
        /// </summary>
        public static ModelsApi Models { get; } = new ModelsApi();

        /// <summary>
        /// Gets an instance of the LayersApi class, which provides access to a set of pre-defined neural network layers.
        /// </summary>
        public static LayersApi Layers { get; } = new LayersApi();

        /// <summary>
        /// Gets an instance of the ActivationsApi class, which provides access to a collection of pre-defined neural network activation functions.
        /// </summary>
        public static ActivationsApi Activations { get; } = new ActivationsApi();

        /// <summary>
        /// Gets an instance of the RandomApi class, which provides access to random number generation functionalities.
        /// </summary>
        public static RandomApi Random { get; } = new RandomApi();

        /// <summary>
        /// Gets an instance of the InitializersApi class, which provides access to various weight and bias initialization methods for neural network components.
        /// </summary>
        public static InitializersApi Initializers { get; } = new InitializersApi();

        /// <summary>
        /// Gets an instance of the SizeApi class, which provides access to size objects in different dimensions.
        /// </summary>
        public static SizeApi Size { get; } = new SizeApi();
    }
}
