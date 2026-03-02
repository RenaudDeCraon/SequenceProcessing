package SequenceProcessing.Functions;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import Math.Tensor;

import java.io.Serializable;
import java.util.ArrayList;

public class Transpose implements ComputationalGraph.Function.Function, Serializable {

    @Override
    public Tensor calculate(Tensor tensor) {
        return tensor.transpose(new int[]{1, 0});
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        return backward.transpose(new int[]{1, 0});
    }

    @Override
    public ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased) {
        ComputationalNode newNode = new FunctionNode(isBiased, this);
        inputNodes.get(0).add(newNode);
        return newNode;
    }
}
