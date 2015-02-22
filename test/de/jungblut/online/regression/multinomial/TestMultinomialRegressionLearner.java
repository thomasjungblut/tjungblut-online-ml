package de.jungblut.online.regression.multinomial;

import java.util.List;
import java.util.function.IntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.random.RandomDataImpl;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.activation.SigmoidActivationFunction;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.squashing.LogisticErrorFunction;
import de.jungblut.online.minimizer.StochasticGradientDescent;
import de.jungblut.online.minimizer.StochasticGradientDescent.StochasticGradientDescentBuilder;
import de.jungblut.online.ml.FeatureOutcomePair;
import de.jungblut.online.regression.RegressionLearner;

public class TestMultinomialRegressionLearner {

  private RandomDataImpl rnd;

  @Before
  public void setup() {
    rnd = new RandomDataImpl();
    rnd.reSeed(0);
  }

  @Test
  public void testSimpleMultinomialRegression() {
    IntFunction<RegressionLearner> factory = (i) -> {
      StochasticGradientDescent minimizer = StochasticGradientDescentBuilder
          .create(1e-4).progressReportInterval(100_000).build();
      RegressionLearner learner = new RegressionLearner(minimizer,
          new SigmoidActivationFunction(), new LogisticErrorFunction());
      learner.setNumPasses(50);
      return learner;
    };

    MultinomialRegressionLearner learner = new MultinomialRegressionLearner(
        factory);

    List<FeatureOutcomePair> trainingSet = generateData();

    MultinomialRegressionModel model = learner
        .train(() -> trainingSet.stream());

    double acc = computeClassificationAccuracy(generateData(), model);
    Assert.assertEquals(1d, acc, 0.1);
  }

  public double computeClassificationAccuracy(List<FeatureOutcomePair> data,
      MultinomialRegressionModel model) {

    double correct = 0;
    MultinomialRegressionClassifier clf = new MultinomialRegressionClassifier(
        model);
    for (FeatureOutcomePair pair : data) {
      DoubleVector prediction = clf.predict(pair.getFeature());
      if (prediction.maxIndex() == pair.getOutcome().maxIndex()) {
        correct++;
      }
    }
    return correct / data.size();
  }

  public List<FeatureOutcomePair> generateData() {
    // similar to the mickey mouse data set
    final int[] centersX = new int[] { 25, 50, 75 };
    final int[] centersY = new int[] { 25, 150, 75 };
    return IntStream
        .range(1, 5000)
        .mapToObj(
            (i) -> {
              int clz = i % centersX.length;
              double meanX = centersX[clz];
              double meanY = centersY[clz];
              double stddev = 5d;
              double[] feat = new double[] { 1,
                  rnd.nextGaussian(meanX, stddev),
                  rnd.nextGaussian(meanY, stddev) };

              DoubleVector outcome = new DenseDoubleVector(centersX.length);
              outcome.set(clz, 1d);
              return new FeatureOutcomePair(new DenseDoubleVector(feat),
                  outcome);
            }).collect(Collectors.toList());
  }
}
