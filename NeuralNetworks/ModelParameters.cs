using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class ModelParameters
    {
        private readonly IList<float> values;
        private readonly bool isExtendable;
        private int allocationPointer;

        public IList<float> Values => values;
        public int Count => values.Count;

        public ModelParameters()
        {
            values = new List<float>();
            isExtendable = true;
            allocationPointer = 0;
        }

        public ModelParameters(IEnumerable<float> values)
        {
            if (values == null) throw new NullReferenceException(nameof(values));

            this.values = values.ToList();
            isExtendable = false;
            allocationPointer = 0;
        }

        public ParameterRange AllocateRange(int size)
        {
            if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Must be bigger than 0");

            if (isExtendable)
            {
                for (int i = 0; i < size; i++)
                {
                    values.Add(0);
                }
            }
            var rangeStartIndex = allocationPointer;
            allocationPointer += size;

            return new ParameterRange(rangeStartIndex, size, values);
        }
    }

    public class ParameterRange : IEnumerable<float>
    {
        private readonly IList<float> values;
        private readonly int start;
        public int Size { get; }

        public ParameterRange(int start, int size, IList<float> values)
        {
            if (start < 0) throw new ArgumentOutOfRangeException(nameof(start), "Cannot be negative");
            if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Must be bigger than 0");
            if (values == null) throw new NullReferenceException(nameof(values));
            if (values.Count < start + size) throw new ArgumentException("Value list is smaller than the range's end index");

            Size = size;
            this.start = start;
            this.values = values;
        }

        public float this[int index]
        {
            get
            {
                if (index >= Size || index < 0) throw new IndexOutOfRangeException($"Index {index} is out of range");
                return values[start + index];
            }
            set
            {
                if (index >= Size || index < 0) throw new IndexOutOfRangeException($"Index {index} is out of range");
                values[start + index] = value;
            }
        }

        public IEnumerator<float> GetEnumerator()
        {
            for (int i = 0; i < Size; i++)
            {
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void CopyTo(float[] values)
        {
            for (int i = 0; i < Size; i++)
            {
                values[i] = this[i];
            }
        }

        public float[] ToArray()
        {
            var result = new float[Size];
            CopyTo(result);
            return result;
        }

        public static ParameterRange Of(params float[] values)
        {
            return new ParameterRange(0, values.Length, values);
        }
    }
}
