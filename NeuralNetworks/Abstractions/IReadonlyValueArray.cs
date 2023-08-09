using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks.Abstractions
{
    public interface IReadonlyValueArray : IEnumerable<float>
    {
        public float this[int index]
        {
            get;
        }

        public int Count { get; }
    }
}
