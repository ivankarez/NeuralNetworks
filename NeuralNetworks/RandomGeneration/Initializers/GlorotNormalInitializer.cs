﻿using Ivankarez.NeuralNetworks.Abstractions;
using System;

namespace Ivankarez.NeuralNetworks.RandomGeneration.Initializers
{
    public class GlorotNormalInitializer : IInitializer
    {
        public IRandomProvider RandomProvider { get; }

        public GlorotNormalInitializer(IRandomProvider randomProvider) 
        {
            if (randomProvider == null) throw new ArgumentNullException(nameof(randomProvider));

            RandomProvider = randomProvider;
        }

        public float GenerateValue(int fanIn, int fanOut)
        {
            var stdDev = (float)Math.Sqrt(2.0 / (fanIn + fanOut));

            return RandomProvider.NextNormalFloat(0, stdDev);
        }
    }
}
