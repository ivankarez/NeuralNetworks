using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Api
{
    public class ModelsApi
    {
        internal ModelsApi()
        { }

        /// <summary>
        /// Creates and returns a Layered Network Model, a composite neural network model consisting of multiple layers, with the specified number of input nodes and layers.
        /// </summary>
        /// <param name="inputs">The number of input nodes to the network model.</param>
        /// <param name="layers">An array of IModelLayer instances representing the layers of the network model.</param>
        public LayeredNetworkModel Layered(int inputs, params IModelLayer[] layers)
        {
            return new LayeredNetworkModel(inputs, layers);
        }
    }
}
