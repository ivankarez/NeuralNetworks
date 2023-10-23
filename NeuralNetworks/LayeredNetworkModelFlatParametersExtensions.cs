using Ivankarez.NeuralNetworks.Utils;
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
            var result = new List<float>();
            foreach (var layer in model.Layers)
            {
                foreach (var paramName in layer.Parameters.Get1dVectorNames())
                {
                    result.AddRange(layer.Parameters.Get1dVector(paramName));
                }

                foreach (var paramName in layer.Parameters.Get2dVectorNames())
                {
                    result.AddRange(layer.Parameters.Get2dVector(paramName));
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
            var flatIndex = 0;
            foreach (var layer in model.Layers)
            {
                foreach (var paramName in layer.Parameters.Get1dVectorNames())
                {
                    var vector = layer.Parameters.Get1dVector(paramName);
                    for (int i = 0; i < vector.Length; i++)
                    {
                        vector[i] = flatParameters[flatIndex++];
                    }
                }

                foreach (var paramName in layer.Parameters.Get2dVectorNames())
                {
                    var vector = layer.Parameters.Get2dVector(paramName);
                    for (int x = 0; x < vector.GetLength(0); x++)
                    {
                        for (int y = 0; y < vector.GetLength(1); y++)
                        {
                            vector[x, y] = flatParameters[flatIndex++];
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Counts the total number of trainable parameters in a LayeredNetworkModel.
        /// This includes both 1D and 2D vectors from all layers in the model.
        /// </summary>
        /// <param name="model">The LayeredNetworkModel to count parameters in.</param>
        /// <returns>The total count of trainable parameters in the model.</returns>
        public static int CountParameters(this LayeredNetworkModel model)
        {
            var count = 0;
            foreach (var layer in model.Layers)
            {
                foreach (var paramName in layer.Parameters.Get1dVectorNames())
                {
                    count += layer.Parameters.Get1dVector(paramName).Length;
                }

                foreach (var paramName in layer.Parameters.Get2dVectorNames())
                {
                    count += layer.Parameters.Get2dVector(paramName).Length;
                }
            }

            return count;
        }
    }
}
