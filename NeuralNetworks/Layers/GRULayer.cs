using Ivankarez.NeuralNetworks.Abstractions;
using Ivankarez.NeuralNetworks.Utils;
using Ivankarez.NeuralNetworks.Values;
using System;

namespace Ivankarez.NeuralNetworks.Layers
{
    public class GRULayer : IModelLayer
    {
        public int NodeCount { get; }

        private readonly IActivation gateActivation;
        private readonly IActivation candidateActivation;
        private readonly bool useBias;

        private float[] updateGateInputs;
        private float[] resetGateInputs;
        private float[] candidateInputs;

        private ValueStoreRange[] updateGateWeights;
        private ValueStoreRange[] resetGateWeights;
        private ValueStoreRange[] candidateWeights;

        private ValueStoreRange recurrentUpdateGateWeights;
        private ValueStoreRange recurrentResetGateWeights;
        private ValueStoreRange recurrentCandidateWeights;

        private ValueStoreRange kernels;

        public GRULayer(int nodeCount, IActivation gateActivation, IActivation candidateActivation, bool useBias)
        {
            if (nodeCount <= 0) throw new ArgumentOutOfRangeException(nameof(nodeCount), "Must be bigger than zero");
            if (gateActivation == null) throw new ArgumentNullException(nameof(gateActivation));
            if (candidateActivation == null) throw new ArgumentNullException(nameof(candidateActivation));

            NodeCount = nodeCount;
            this.gateActivation = gateActivation;
            this.candidateActivation = candidateActivation;
            this.useBias = useBias;
        }

        public void Build(int inputSize, ValueStore parameters, ValueStore state)
        {
            var nodeInputSize = useBias ? inputSize + 1 : inputSize;
            updateGateWeights = new ValueStoreRange[NodeCount];
            resetGateWeights = new ValueStoreRange[NodeCount];
            candidateWeights = new ValueStoreRange[NodeCount];

            for (int i = 0; i < NodeCount; i++)
            {
                updateGateWeights[i] = parameters.AllocateRange(nodeInputSize);
                resetGateWeights[i] = parameters.AllocateRange(nodeInputSize);
                candidateWeights[i] = parameters.AllocateRange(nodeInputSize);
            }

            recurrentUpdateGateWeights = parameters.AllocateRange(NodeCount);
            recurrentResetGateWeights = parameters.AllocateRange(NodeCount);
            recurrentCandidateWeights = parameters.AllocateRange(NodeCount);

            updateGateInputs = new float[nodeInputSize + 1];
            resetGateInputs = new float[nodeInputSize + 1];
            candidateInputs = new float[nodeInputSize + 1];

            kernels = state.AllocateRange(NodeCount);
        }

        public IValueArray Update(IValueArray inputValues)
        {
            for (int nodeIndex = 0; nodeIndex < NodeCount; nodeIndex++)
            {
                var updateGateWeight = updateGateWeights[nodeIndex];
                var resetGateWeight = resetGateWeights[nodeIndex];
                var candidateWeight = candidateWeights[nodeIndex];

                MathUtils.ElementwiseMultiply(inputValues, updateGateWeight, updateGateInputs);
                MathUtils.ElementwiseMultiply(inputValues, resetGateWeight, resetGateInputs);

                if (useBias)
                {
                    SetBiasInput(updateGateWeight[^1], resetGateWeight[^1], candidateWeight[^1]);
                }

                SetRecurrentInput(updateGateInputs, recurrentUpdateGateWeights[nodeIndex] * kernels[nodeIndex]);
                SetRecurrentInput(resetGateInputs, recurrentResetGateWeights[nodeIndex] * kernels[nodeIndex]);

                var updateGate = gateActivation.Apply(updateGateInputs);
                var resetGate = gateActivation.Apply(resetGateInputs);

                var resetHidden = kernels[nodeIndex] * resetGate;
                SetRecurrentInput(candidateInputs, resetHidden * recurrentCandidateWeights[nodeIndex]);
                MathUtils.ElementwiseMultiply(inputValues, candidateWeight, candidateInputs);

                var candidate = candidateActivation.Apply(candidateInputs);

                kernels[nodeIndex] = (1 - updateGate) * candidate + updateGate * resetHidden;
            }

            return kernels;
        }

        private static void SetRecurrentInput(float[] inputs, float value)
        {
            inputs[^1] = value;
        }

        private void SetBiasInput(float updateGateBias, float resetGateBias, float candidateBias)
        {
            updateGateInputs[^2] = updateGateBias;
            resetGateInputs[^2] = resetGateBias;
            candidateInputs[^2] = candidateBias;
        }
    }
}
