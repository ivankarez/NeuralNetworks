using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Api
{
    public class ModelsApi
    {
        internal ModelsApi()
        { }

        /// <summary>
        /// Creates a new layered network model.
        /// </summary>
        /// <param name="inputs">Size of the input vector.</param>
        /// <param name="layers">Layers of the network in order.</param>
        /// <returns>The created model</returns>
        public LayeredNetworkModel Layered(int inputs, params IModelLayer[] layers)
        {
            return new LayeredNetworkModel(inputs, layers);
        }
    }
}
