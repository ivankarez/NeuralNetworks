using System.Collections.Generic;

namespace Ivankarez.NeuralNetworks.Values
{
    public class NamedVectors<T>
    {
        private readonly IDictionary<string, T[]> vectors1d;
        private readonly IDictionary<string, T[,]> vectors2d;

        public NamedVectors()
        {
            vectors1d = new Dictionary<string, T[]>();
            vectors2d = new Dictionary<string, T[,]>();
        }

        public void Add(string name, T[] vector)
        {
            vectors1d.Add(name, vector);
        }

        public T[] Get1dVector(string name)
        {
            return vectors1d[name];
        }

        public void Add(string name, T[,] vector)
        {
            vectors2d.Add(name, vector);
        }

        public T[,] Get2dVector(string name)
        {
            return vectors2d[name];
        }

        public ICollection<string> Get1dVectorNames()
        {
            return vectors1d.Keys;
        }

        public ICollection<string> Get2dVectorNames()
        {
            return vectors2d.Keys;
        }
    }
}
