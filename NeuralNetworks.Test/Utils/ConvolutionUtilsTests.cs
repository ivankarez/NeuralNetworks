using FluentAssertions;
using Ivankarez.NeuralNetworks.Utils;
using NUnit.Framework;
using System;

namespace Ivankarez.NeuralNetworks.Test.Utils
{
    public class ConvolutionUtilsTests
    {
        [TestCase(3, 1, 1, 3)]
        [TestCase(3, 2, 1, 2)]
        [TestCase(3, 3, 1, 1)]
        [TestCase(3, 1, 2, 2)]
        [TestCase(3, 2, 2, 1)]
        [TestCase(3, 3, 2, 1)]
        [TestCase(3, 1, 3, 1)]
        public void TestCalculateOutputSize_HappyPath(int inputSize, int window, int stride, int expectedOutputSize)
        {
            var outputSize = ConvolutionUtils.CalculateOutputSize(inputSize, window, stride);
            outputSize.Should().Be(expectedOutputSize);
        }

        [TestCase(0)]
        [TestCase(-1)]
        public void TestCalculateOutputSize_InvalidInputSize(int inputSize)
        {
            Action testAction = () => ConvolutionUtils.CalculateOutputSize(inputSize, 1, 1);
            testAction.Should().Throw<ArgumentException>().WithMessage("Input size must be greater than 0*");
        }

        [TestCase(0)]
        [TestCase(-1)]
        public void TestCalculateOutputSize_InvalidWindow(int window)
        {
            Action testAction = () => ConvolutionUtils.CalculateOutputSize(1, window, 1);
            testAction.Should().Throw<ArgumentException>().WithMessage("Window size must be greater than 0*");
        }

        [TestCase(0)]
        [TestCase(-1)]
        public void TestCalculateOutputSize_InvalidStride(int stride)
        {
            Action testAction = () => ConvolutionUtils.CalculateOutputSize(1, 1, stride);
            testAction.Should().Throw<ArgumentException>().WithMessage("Stride must be greater than 0*");
        }

        [TestCase(1, 2)]
        [TestCase(2, 3)]
        public void TestCalculateOutputSize_InvalidWindowGreaterThanInputSize(int inputSize, int window)
        {
            Action testAction = () => ConvolutionUtils.CalculateOutputSize(inputSize, window, 1);
            testAction.Should().Throw<ArgumentException>().WithMessage("Window size must be less than input size*");
        }
    }
}
