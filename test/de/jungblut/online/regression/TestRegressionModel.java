package de.jungblut.online.regression;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.activation.SigmoidActivationFunction;
import de.jungblut.math.dense.DenseDoubleVector;

public class TestRegressionModel {

  @Test
  public void testSerDe() throws IOException {
    DoubleVector weights = new DenseDoubleVector(new double[] { 1, 2, 3, 4, 5 });
    SigmoidActivationFunction activation = new SigmoidActivationFunction();
    RegressionModel model = new RegressionModel(weights, activation);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    DataOutputStream dos = new DataOutputStream(baos);
    model.serialize(dos);

    ByteArrayInputStream bais = new ByteArrayInputStream(baos.toByteArray());
    DataInputStream dis = new DataInputStream(bais);
    model = new RegressionModel();
    RegressionModel deserialized = model.deserialize(dis);

    Assert.assertArrayEquals(weights.toArray(), deserialized.getWeights()
        .toArray(), 1e-8);

    Assert.assertEquals(activation.getClass(), model.getActivationFunction()
        .getClass());
  }

}
