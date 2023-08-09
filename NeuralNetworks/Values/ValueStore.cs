using Ivankarez.NeuralNetworks.Abstractions;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ivankarez.NeuralNetworks.Values
{
    public class ValueStore
    {
        private readonly ValueArray values;
        private int allocationPointer;

        public IValueArray Values => values;
        public int Count => values.Count;

        public ValueStore()
        {
            values = new ValueArray();
            allocationPointer = 0;
        }

        public ValueStore(IEnumerable<float> values)
        {
            if (values == null) throw new NullReferenceException(nameof(values));

            this.values = new ValueArray(values.ToArray());
            allocationPointer = 0;
        }

        internal ValueStoreRange AllocateRange(int size)
        {
            if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Must be bigger than 0");

            for (int i = 0; i < size; i++)
            {
                values.Extend(0);
            }
            var rangeStartIndex = allocationPointer;
            allocationPointer += size;

            return new ValueStoreRange(rangeStartIndex, size, values);
        }
    }
}
