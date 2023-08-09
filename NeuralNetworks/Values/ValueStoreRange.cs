using Ivankarez.NeuralNetworks.Abstractions;
using System;
using System.Collections;
using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks.Values
{
    public class ValueStoreRange : IValueArray
    {
        private readonly IValueArray values;
        private readonly int start;

        public int Count { get; }

        public ValueStoreRange(int start, int count, IValueArray values)
        {
            if (start < 0) throw new ArgumentOutOfRangeException(nameof(start), "Cannot be negative");
            if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count), "Must be bigger than 0");
            if (values == null) throw new NullReferenceException(nameof(values));
            if (values.Count < start + count) throw new ArgumentException("Value list is smaller than the range's end index");

            Count = count;
            this.start = start;
            this.values = values;
        }

        public float this[int index]
        {
            get
            {
                if (index >= Count || index < 0) throw new IndexOutOfRangeException($"Index {index} is out of range");
                return values[start + index];
            }
            set
            {
                if (index >= Count || index < 0) throw new IndexOutOfRangeException($"Index {index} is out of range");
                values[start + index] = value;
            }
        }

        public IEnumerator<float> GetEnumerator()
        {
            for (int i = 0; i < Count; i++)
            {
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public static ValueStoreRange Of(params float[] values)
        {
            var valueArray = new ValueArray(values);
            return new ValueStoreRange(0, values.Length, valueArray);
        }
    }
}
