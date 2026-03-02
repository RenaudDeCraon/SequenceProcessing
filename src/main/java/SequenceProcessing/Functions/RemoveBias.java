package SequenceProcessing.Functions;

import java.io.Serializable;
import java.util.ArrayList;

import ComputationalGraph.Node.ComputationalNode;
import ComputationalGraph.Node.FunctionNode;
import Math.Tensor;

public class RemoveBias implements ComputationalGraph.Function.Function, Serializable {
    @Override
    public Tensor calculate(Tensor matrix) {
        ArrayList<Double> data = (ArrayList<Double>) matrix.getData();
        ArrayList<Double> values = new ArrayList<>();
        for (int i = 0; i < data.size() - 1; i++) {
            values.add(data.get(i));
        }
        return new Tensor(values, new int[]{1, values.size()});
    }

    @Override
    public Tensor derivative(Tensor value, Tensor backward) {
        ArrayList<Double> values = (ArrayList<Double>) backward.getData();
        ArrayList<Double> newValues = new ArrayList<>(values);
        newValues.add(0.0);
        return new Tensor(newValues, new int[]{1, newValues.size()});
    }

    @Override
    public ComputationalNode addEdge(ArrayList<ComputationalNode> inputNodes, boolean isBiased) {
        ComputationalNode newNode = new FunctionNode(isBiased, this);
        inputNodes.get(0).add(newNode);
        return newNode;
    }
}
