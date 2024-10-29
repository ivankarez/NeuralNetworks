using System.Collections.ObjectModel;

namespace Ivankarez.NeuralNetworks.Abstractions
{
    internal interface ITensor
    {
        public int Rank { get; }
        public string Name { get; set; }
        public ReadOnlyCollection<int> Shape { get; }
    }
}
