using Ivankarez.NeuralNetworks.Activations;

namespace Ivankarez.NeuralNetworks.Api
{
    public class ActivationsApi
    {
        internal ActivationsApi() { }

        /// <summary>
        /// Creates and returns a Linear Activation function, which is a basic activation function that outputs the input as is.
        /// </summary>
        /// <returns>A Linear Activation instance.</returns>
        public LinearActivation Linear() => new LinearActivation();

        /// <summary>
        /// Creates and returns a Clamped Linear Activation function with optional minimum and maximum bounds.
        /// </summary>
        /// <param name="min">The lower bound for clamping the output (default is -1).</param>
        /// <param name="max">The upper bound for clamping the output (default is 1).</param>
        /// <returns>A Clamped Linear Activation instance with the specified or default bounds.</returns>
        public ClampedLinearActivation ClampedLinear(int min = -1, int max = 1) => new ClampedLinearActivation(min, max);

        /// <summary>
        /// Creates and returns a Hyperbolic Tangent (Tanh) Activation function, which is a commonly used activation function in neural networks.
        /// </summary>
        /// <returns>A Tanh Activation instance.</returns>
        public TanhActivation Tanh() => new TanhActivation();

        /// <summary>
        /// Creates and returns a Sigmoid Activation function, which is a widely used activation function in neural networks for introducing non-linearity.
        /// </summary>
        /// <returns>A Sigmoid Activation instance.</returns>
        public SigmoidActivation Sigmoid() => new SigmoidActivation();

        /// <summary>
        /// Creates and returns a Rectified Linear Unit (ReLU) Activation function, a popular choice for introducing non-linearity in neural networks.
        /// </summary>
        /// <returns>A ReLU Activation instance.</returns>
        public ReluActivation Relu() => new ReluActivation();

        /// <summary>
        /// Creates and returns a Leaky Rectified Linear Unit (Leaky ReLU) Activation function with an optional alpha parameter.
        /// </summary>
        /// <param name="alpha">The slope of the negative part of the activation (default is 0.3f).</param>
        /// <returns>A Leaky ReLU Activation instance with the specified or default alpha value.</returns>
        public LeakyReluActivation LeakyRelu(float alpha = .3f) => new LeakyReluActivation(alpha);
    }
}
