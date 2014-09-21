package de.jungblut.online.bayes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import de.jungblut.math.DoubleMatrix;
import de.jungblut.math.DoubleVector;
import de.jungblut.online.ml.Model;
import de.jungblut.writable.MatrixWritable;
import de.jungblut.writable.VectorWritable;

public class BayesianProbabilityModel implements Model {

  private DoubleMatrix probabilityMatrix;
  private DoubleVector classPriorProbability;

  // deserialization constructor
  public BayesianProbabilityModel() {
  }

  public BayesianProbabilityModel(DoubleMatrix probabilityMatrix,
      DoubleVector classPriorProbability) {
    this.probabilityMatrix = probabilityMatrix;
    this.classPriorProbability = classPriorProbability;
  }

  @Override
  public void serialize(DataOutput out) throws IOException {
    new MatrixWritable(probabilityMatrix).write(out);
    VectorWritable.writeVector(classPriorProbability, out);
  }

  @Override
  public BayesianProbabilityModel deserialize(DataInput in) throws IOException {
    MatrixWritable mat = new MatrixWritable();
    mat.readFields(in);
    this.probabilityMatrix = mat.getMatrix();
    this.classPriorProbability = VectorWritable.readVector(in);
    return this;
  }

  public DoubleVector getClassPriorProbability() {
    return this.classPriorProbability;
  }

  public DoubleMatrix getProbabilityMatrix() {
    return this.probabilityMatrix;
  }

}
