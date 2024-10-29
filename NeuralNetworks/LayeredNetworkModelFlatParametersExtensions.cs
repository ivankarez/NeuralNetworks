using Ivankarez.NeuralNetworks.Layers;
using Ivankarez.NeuralNetworks.Utils;
using System;
using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks
{
    public static class LayeredNetworkModelFlatParametersExtensions
    {
        /// <summary>
        /// Retrieves the flattened array of all trainable parameters in a LayeredNetworkModel.
        /// This includes both 1D and 2D vectors from all layers in the model.
        /// </summary>
        /// <param name="model">The LayeredNetworkModel to extract parameters from.</param>
        /// <returns>An array containing all the flattened trainable parameters.</returns>
        public static float[] GetParametersFlat(this LayeredNetworkModel model)
        {
            // TODO: Implement other types of layers
            var result = new List<float>();
            foreach (var layer in model.Layers)
            {
                if (layer is DenseLayer)
                {
                    var denseLayer = layer as DenseLayer;
                    result.AddRange(denseLayer.Weights);
                    if (denseLayer.Biases != null)
                    {
                        result.AddRange(denseLayer.Biases);
                    }
                } 
                else
                {
                    throw new NotImplementedException($"GetParametersFlat doesn't support layer of type {layer.GetType().Name}");
                }
            }

            return result.ToArray();
        }

        /// <summary>
        /// Sets the parameters of a LayeredNetworkModel using a flattened array of parameters.
        /// This method assigns the values from the flat array to the corresponding parameters
        /// in each layer of the model, considering both 1D and 2D vectors.
        /// </summary>
        /// <param name="model">The LayeredNetworkModel to update with the flat parameters.</param>
        /// <param name="flatParameters">The flattened array of parameters to assign to the model.</param>
        public static void SetParametersFlat(this LayeredNetworkModel model, float[] flatParameters)
        {
            throw new NotImplementedException("SetParametersFlat is not implemented yet");
        }

        /// <summary>
        /// Counts the total number of trainable parameters in a LayeredNetworkModel.
        /// This includes both 1D and 2D vectors from all layers in the model.
        /// </summary>
        /// <param name="model">The LayeredNetworkModel to count parameters in.</param>
        /// <returns>The total count of trainable parameters in the model.</returns>
        public static int CountParameters(this LayeredNetworkModel model)
        {
            throw new NotImplementedException("CountParameters is not implemented yet");
        }
    }
}
