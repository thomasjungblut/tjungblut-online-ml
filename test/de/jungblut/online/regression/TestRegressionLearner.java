package de.jungblut.online.regression;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.random.RandomDataImpl;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.MathUtils;
import de.jungblut.math.activation.SigmoidActivationFunction;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.dense.SingleEntryDoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.math.squashing.LogisticErrorFunction;
import de.jungblut.online.minimizer.StochasticGradientDescent;
import de.jungblut.online.minimizer.StochasticGradientDescent.StochasticGradientDescentBuilder;
import de.jungblut.online.ml.FeatureOutcomePair;

public class TestRegressionLearner {

  private RandomDataImpl rnd;

  @Before
  public void setup() {
    rnd = new RandomDataImpl();
    rnd.reSeed(0);
  }

  @Test
  public void gradCheck() {
    RegressionLearner learner = new RegressionLearner(
        StochasticGradientDescentBuilder.create(0.1).build(),
        new SigmoidActivationFunction(), new LogisticErrorFunction());
    learner.setRandom(new Random(0));

    // for both classes
    for (int i = 0; i <= 1; i++) {
      // 0.1 steps between zero and 1
      for (double d = 0.0; d <= 1.0; d += 0.1) {
        DoubleVector weights = new DenseDoubleVector(new double[] { d, d });

        DoubleVector nextFeature = new DenseDoubleVector(new double[] { 1, 10 });
        DoubleVector nextOutcome = new DenseDoubleVector(new double[] { i });
        DoubleVector numGrad = MathUtils.numericalGradient(weights,
            (x) -> learner.observeExample(new FeatureOutcomePair(nextFeature,
                nextOutcome), x));

        CostGradientTuple realGrad = learner.observeExample(
            new FeatureOutcomePair(nextFeature, nextOutcome), weights);
        Assert.assertArrayEquals(numGrad.toArray(), realGrad.getGradient()
            .toArray(), 1e-4);
      }
    }
  }

  @Test
  public void testSimpleLogisticRegression() {
    List<FeatureOutcomePair> data = generateData();

    RegressionLearner learner = newLearner();

    RegressionModel model = learner.train(() -> data.stream());
    Assert.assertArrayEquals(new double[] { -159.7796434436107,
        1.178953822695672, 2.0180958310781554 }, model.getWeights().toArray(),
        1e-4);
    double acc = computeClassificationAccuracy(generateData(), model);
    Assert.assertEquals(1d, acc, 1e-3);
  }

  @Test
  public void testParallelLogisticRegression() {
    List<FeatureOutcomePair> data = generateData();
    RegressionLearner learner = newLearner();

    // the parallel one converges usually to a different version, because there
    // is no defined order of updates, thus we only assert the accuracy.
    RegressionModel model = learner.train(() -> data.stream().parallel());
    double acc = computeClassificationAccuracy(generateData(), model);
    Assert.assertEquals(1d, acc, 1e-3);
  }

  public RegressionLearner newLearner() {
    StochasticGradientDescent min = StochasticGradientDescentBuilder
        .create(0.1).build();
    RegressionLearner learner = new RegressionLearner(min,
        new SigmoidActivationFunction(), new LogisticErrorFunction());
    learner.setRandom(new Random(1337));
    learner.setNumPasses(25);
    return learner;
  }

  public double computeClassificationAccuracy(List<FeatureOutcomePair> data,
      RegressionModel model) {

    double correct = 0;
    RegressionClassifier clf = new RegressionClassifier(model);
    for (FeatureOutcomePair pair : data) {
      DoubleVector prediction = clf.predict(pair.getFeature());
      if (prediction.subtract(pair.getOutcome()).abs().sum() < 0.1d) {
        correct++;
      }
    }
    return correct / data.size();
  }

  public List<FeatureOutcomePair> generateData() {
    return IntStream
        .range(1, 2000)
        .mapToObj(
            (i) -> {
              double mean = i % 2 == 0 ? 25d : 75d;
              double stddev = 10d;
              double clzVal = i % 2 == 0 ? 0d : 1d;
              double[] feat = new double[] { 1, rnd.nextGaussian(mean, stddev),
                  rnd.nextGaussian(mean, stddev) };

              return new FeatureOutcomePair(new DenseDoubleVector(feat),
                  new SingleEntryDoubleVector(clzVal));
            }).collect(Collectors.toList());
  }
}
