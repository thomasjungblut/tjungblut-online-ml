package de.jungblut.online.regression;

import org.junit.Assert;
import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.activation.SigmoidActivationFunction;
import de.jungblut.math.dense.DenseDoubleVector;

public class TestRegressionClassifier {

  @Test
  public void testClassifier() {
    // weights from the learner test
    DenseDoubleVector weights = new DenseDoubleVector(new double[] {
        -159.7796434436107, 1.178953822695672, 2.0180958310781554 });

    RegressionClassifier classifier = new RegressionClassifier(weights,
        new SigmoidActivationFunction());

    DoubleVector prediction = classifier.predict(new DenseDoubleVector(
        new double[] { 1, 75d, 75d }));
    Assert.assertEquals(1d, prediction.get(0), 1e-4);

    prediction = classifier.predict(new DenseDoubleVector(new double[] { 1,
        25d, 25d }));
    Assert.assertEquals(0d, prediction.get(0), 1e-4);
  }
}
