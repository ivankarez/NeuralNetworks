using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.RandomGeneration.Initializers;

namespace Ivankarez.NeuralNetworks.Api
{
    public class InitializersApi
    {
        internal InitializersApi() { }

        /// <summary>
        /// Creates and returns a Zeros Initializer, which is used to initialize values with zeros in various components of a neural network.
        /// </summary>
        /// <returns>A Zeros Initializer instance.</returns>
        public ZerosInitializer Zeros() => new ZerosInitializer();

        /// <summary>
        /// Creates and returns a Constant Initializer, which sets the initial value of components to a specified constant.
        /// </summary>
        /// <param name="value">The constant value to which components should be initialized.</param>
        /// <returns>A Constant Initializer instance with the specified constant value.</returns>
        public ConstantInitializer Constant(float value) => new ConstantInitializer(value);

        /// <summary>
        /// Creates and returns a Uniform Initializer for initializing values within a specified range, using the provided or default random number generator.
        /// </summary>
        /// <param name="min">The minimum value for initialization (defaults to -1.0).</param>
        /// <param name="max">The maximum value for initialization (defaults to 1.0).</param>
        /// <param name="randomProvider">The random number provider to use for generating values (defaults to the default system Random).</param>
        /// <returns>A Uniform Initializer instance for the specified range and random number provider.</returns>
        public UniformInitializer Uniform(float min = -1f, float max = 1f, IRandomProvider randomProvider = null)
        {
            randomProvider ??= NN.Random.Default();

            return new UniformInitializer(randomProvider, min, max);
        }

        /// <summary>
        /// Creates and returns a Normal Initializer for initializing values with a Gaussian distribution, using the provided or default random number generator.
        /// </summary>
        /// <param name="mean">The mean (average) of the Gaussian distribution (defaults to 0.0).</param>
        /// <param name="stdDev">The standard deviation of the Gaussian distribution (defaults to 0.05).</param>
        /// <param name="randomProvider">The random number provider to use for generating values (defaults to the default system Random).</param>
        /// <returns>A Normal Initializer instance for the specified mean, standard deviation, and random number provider.</returns>
        public NormalInitializer Normal(float mean = 0f, float stdDev = .05f, IRandomProvider randomProvider = null)
        {
            randomProvider ??= NN.Random.Default();

            return new NormalInitializer(randomProvider, mean, stdDev);
        }

        /// <summary>
        /// Creates and returns a Glorot (Xavier) Uniform Initializer for initializing values according to the Glorot initialization scheme, using the provided or default random number generator.
        /// </summary>
        /// <param name="randomProvider">The random number provider to use for generating values (defaults to the default system Random).</param>
        /// <returns>A Glorot Uniform Initializer instance with the specified or default random number provider.</returns>
        public GlorotUniformInitializer GlorotUniform(IRandomProvider randomProvider = null)
        {
            randomProvider ??= NN.Random.Default();

            return new GlorotUniformInitializer(randomProvider);
        }

        /// <summary>
        /// Creates and returns a Glorot (Xavier) Normal Initializer for initializing values according to the Glorot initialization scheme with a normal (Gaussian) distribution, using the provided or default random number generator.
        /// </summary>
        /// <param name="randomProvider">The random number provider to use for generating values (defaults to the default system Random).</param>
        /// <returns>A Glorot Normal Initializer instance with the specified or default random number provider.</returns>
        public GlorotNormalInitializer GlorotNormal(IRandomProvider randomProvider = null)
        {
            randomProvider ??= NN.Random.Default();

            return new GlorotNormalInitializer(randomProvider);
        }
    }
}
