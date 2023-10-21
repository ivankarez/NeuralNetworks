using Ivankarez.NeuralNetworks.RandomGeneration;
using System;

namespace Ivankarez.NeuralNetworks.Api
{
    public class RandomApi
    {
        private static SystemRandomProvider defaultRandomProvider = null;

        internal RandomApi() { }

        /// <summary>
        /// Provides the default random number generator for the system. If not already created, a new SystemRandomProvider with a new Random instance is generated and returned.
        /// </summary>
        /// <returns>The default SystemRandomProvider instance with a Random number generator.</returns>
        public SystemRandomProvider Default()
        {
            if (defaultRandomProvider == null)
            {
                defaultRandomProvider = new SystemRandomProvider(new Random());
            }

            return defaultRandomProvider;
        }

        /// <summary>
        /// Creates and returns a SystemRandomProvider, which uses the specified or default Random number generator.
        /// </summary>
        /// <param name="random">The Random instance to be used (defaults to the default system Random instance).</param>
        /// <returns>A SystemRandomProvider instance with the specified or default Random generator.</returns>
        public SystemRandomProvider System(Random random = null)
        {
            random ??= Default().Random;

            return new SystemRandomProvider(random);
        }
    }
}
