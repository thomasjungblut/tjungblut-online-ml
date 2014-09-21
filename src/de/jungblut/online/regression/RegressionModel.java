package de.jungblut.online.regression;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.util.ReflectionUtils;

import com.google.common.base.Preconditions;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.activation.ActivationFunction;
import de.jungblut.online.ml.Model;
import de.jungblut.writable.VectorWritable;

public class RegressionModel implements Model {

  private DoubleVector weights;
  private ActivationFunction activationFunction;

  // deserialization constructor
  public RegressionModel() {
  }

  public RegressionModel(DoubleVector weights,
      ActivationFunction activationFunction) {
    this.weights = Preconditions.checkNotNull(weights, "weights");
    this.activationFunction = Preconditions.checkNotNull(activationFunction,
        "activationFunction");
  }

  @Override
  public void serialize(DataOutput out) throws IOException {
    out.writeUTF(activationFunction.getClass().getName());
    VectorWritable.writeVector(weights, out);
  }

  @Override
  public RegressionModel deserialize(DataInput in) throws IOException {
    String clzName = in.readUTF();
    try {
      this.activationFunction = (ActivationFunction) ReflectionUtils
          .newInstance(Class.forName(clzName), null);
    } catch (ClassNotFoundException e) {
      throw new IOException(e);
    }
    weights = VectorWritable.readVector(in);
    return this;
  }

  public DoubleVector getWeights() {
    return this.weights;
  }

  public ActivationFunction getActivationFunction() {
    return this.activationFunction;
  }

}
