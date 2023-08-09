using Ivankarez.NeuralNetworks.Abstractions;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Ivankarez.NeuralNetworks.Values
{
    public class ValueArray : IValueArray
    {
        private readonly IList<float> values;

        public int Count => values.Count;

        public float this[int index]
        {
            get => values[index];
            set => values[index] = value;
        }

        public ValueArray() : this(new List<float>()) { }

        public ValueArray(IEnumerable<float> initialValues)
        {
            values = initialValues.ToList();
        }

        public void Extend(float value)
        {
            values.Add(value);
        }

        public IEnumerator<float> GetEnumerator()
        {
            return values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return values.GetEnumerator();
        }
    }
}
