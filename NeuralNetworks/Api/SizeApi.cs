using Ivankarez.NeuralNetworks.Utils;

namespace Ivankarez.NeuralNetworks.Api
{
    public class SizeApi
    {
        internal SizeApi() { }

        /// <summary>
        /// Creates a new instance of Size1D with the specified size.
        /// </summary>
        /// <param name="size">The size of the 1D dimension. Must be greater than 0.</param>
        /// <returns>A new Size1D object with the specified size.</returns>
        public Size1D Of(int size)
        {
            return new Size1D(size);
        }

        /// <summary>
        /// Creates a new instance of Size2D with the specified width and height.
        /// </summary>
        /// <param name="width">The width of the 2D size. Must be greater than 0.</param>
        /// <param name="height">The height of the 2D size. Must be greater than 0.</param>
        /// <returns>A new Size2D object with the specified width and height.</returns>
        public Size2D Of(int width, int height)
        {
            return new Size2D(width, height);
        }
    }
}
