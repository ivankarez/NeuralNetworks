using Ivankarez.NeuralNetworks.Abstractions;

namespace Ivankarez.NeuralNetworks.Test.TestUtils
{
    internal class FakeRandomProvider : IRandomProvider
    {
        private readonly float[] fakeResults;
        private int index = 0;

        public FakeRandomProvider(params float[] fakeResults)
        {
            this.fakeResults = fakeResults;
        }

        public float NextFloat()
        {
            var result = fakeResults[index];
            index = (index + 1) % fakeResults.Length;

            return result;
        }
    }
}
