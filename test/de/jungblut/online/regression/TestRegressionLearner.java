package de.jungblut.online.regression;

import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.math3.random.RandomDataImpl;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.MathUtils;
import de.jungblut.math.activation.LinearActivationFunction;
import de.jungblut.math.activation.SigmoidActivationFunction;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.dense.SingleEntryDoubleVector;
import de.jungblut.math.loss.HingeLoss;
import de.jungblut.math.loss.LogLoss;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.online.minimizer.StochasticGradientDescent;
import de.jungblut.online.minimizer.StochasticGradientDescent.StochasticGradientDescentBuilder;
import de.jungblut.online.ml.FeatureOutcomePair;
import de.jungblut.online.regularization.GradientDescentUpdater;
import de.jungblut.online.regularization.L1Regularizer;
import de.jungblut.online.regularization.L2Regularizer;
import de.jungblut.online.regularization.WeightUpdater;

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
        new SigmoidActivationFunction(), new LogLoss());
    learner.setRandom(new Random(0));

    // for both classes
    for (int i = 0; i <= 1; i++) {
      gridGradCheck(learner, i, new GradientDescentUpdater());
    }
  }

  @Test
  public void ridgeGradCheck() {
    RegressionLearner learner = new RegressionLearner(
        StochasticGradientDescentBuilder.create(0.1).build(),
        new SigmoidActivationFunction(), new LogLoss());
    learner.setRandom(new Random(0));

    // for both classes
    for (int i = 0; i <= 1; i++) {
      gridGradCheck(learner, i, new L2Regularizer());
    }
  }

  public void gridGradCheck(RegressionLearner learner, int clz,
      WeightUpdater updater) {
    // 0.1 steps between zero and 1
    for (double d = 0.0; d <= 1.0; d += 0.1) {
      DoubleVector weights = new DenseDoubleVector(new double[] { d, d });

      DoubleVector nextFeature = new DenseDoubleVector(new double[] { 1, 10 });
      DoubleVector nextOutcome = new DenseDoubleVector(new double[] { clz });
      DoubleVector numGrad = MathUtils.numericalGradient(
          weights,
          (x) -> {
            CostGradientTuple tmpGrad = learner.observeExample(
                new FeatureOutcomePair(nextFeature, nextOutcome), x);
            CostGradientTuple tmpUpdatedGradient = updater.computeGradient(x,
                tmpGrad.getGradient(), 1d, 0, 1d, tmpGrad.getCost());

            return new CostGradientTuple(tmpUpdatedGradient.getCost(), null);
          });

      CostGradientTuple realGrad = learner.observeExample(
          new FeatureOutcomePair(nextFeature, nextOutcome), weights);
      // we compute the new weights (to test regularization gradients)
      CostGradientTuple updatedGradient = updater.computeGradient(weights,
          realGrad.getGradient(), 1d, 0, 1d, realGrad.getCost());

      Assert.assertArrayEquals(numGrad.toArray(), updatedGradient.getGradient()
          .toArray(), 1e-4);
    }
  }

  @Test
  public void testSimpleLogisticRegression() {
    List<FeatureOutcomePair> data = generateData();

    RegressionLearner learner = newLearner();

    RegressionModel model = learner.train(() -> data.stream());
    double acc = computeClassificationAccuracy(generateData(), model);
    Assert.assertEquals(1d, acc, 0.1);
  }

  @Test
  public void testLinearSVM() {
    // hinge loss needs -1 vs. 1 as outcome
    double negativeOutputClass = -1;
    List<FeatureOutcomePair> data = generateData(negativeOutputClass);

    RegressionLearner learner = newSVMLearner();

    RegressionModel model = learner.train(() -> data.stream());

    // since SVM's output the distance from the hyperplane to the given feature,
    // we clip the values between -1 and 1.
    // a common technique used to get probabilities is using Platt scaling,
    // which trains a logistic regression on top of the SVM output.
    Function<DoubleVector, DoubleVector> clipping = (v) -> new SingleEntryDoubleVector(
        v.get(0) > 0 ? 1 : negativeOutputClass);

    double acc = computeClassificationAccuracy(
        generateData(negativeOutputClass), model, clipping);
    Assert.assertEquals(1d, acc, 0.1);
  }

  @Test
  public void testRidgeLogisticRegression() {
    List<FeatureOutcomePair> data = generateData();

    RegressionLearner learner = newRegularizedLearner(1d, new L2Regularizer());

    RegressionModel model = learner.train(() -> data.stream());
    double acc = computeClassificationAccuracy(generateData(), model);
    Assert.assertEquals(1d, acc, 0.1);
  }

  @Test
  public void testLassoLogisticRegression() {
    List<FeatureOutcomePair> data = generateDataAddedNoise(2);

    RegressionLearner learner = newRegularizedLearner(1d, new L1Regularizer());

    RegressionModel model = learner.train(() -> data.stream());
    // basically the l1 norm should set the noise to zero
    Assert.assertArrayEquals(new double[] { 0.0, 0.0 }, model.getWeights()
        .sliceByLength(3, 2).toArray(), 1e-2);
    double acc = computeClassificationAccuracy(generateDataAddedNoise(2), model);
    Assert.assertEquals(1d, acc, 0.1);
  }

  @Test
  public void testParallelLogisticRegression() {
    List<FeatureOutcomePair> data = generateData();
    RegressionLearner learner = newLearner();

    // the parallel one converges usually to a different version, because there
    // is no defined order of updates, thus we only assert the accuracy.
    RegressionModel model = learner.train(() -> data.stream().parallel());
    double acc = computeClassificationAccuracy(generateData(), model);
    Assert.assertEquals(1d, acc, 0.1);
  }

  public RegressionLearner newLearner() {
    return newRegularizedLearner(0d, new GradientDescentUpdater());
  }

  public RegressionLearner newRegularizedLearner(double lambda,
      WeightUpdater updater) {
    StochasticGradientDescentBuilder builder = StochasticGradientDescentBuilder
        .create(0.1);
    if (lambda != 0d) {
      builder = builder.lambda(lambda).weightUpdater(updater);
    }
    StochasticGradientDescent min = builder.build();
    RegressionLearner learner = new RegressionLearner(min,
        new SigmoidActivationFunction(), new LogLoss());
    learner.setRandom(new Random(1337));
    learner.setNumPasses(25);
    return learner;
  }

  public RegressionLearner newSVMLearner() {
    StochasticGradientDescentBuilder builder = StochasticGradientDescentBuilder
        .create(0.1);
    StochasticGradientDescent min = builder.build();
    RegressionLearner learner = new RegressionLearner(min,
        new LinearActivationFunction(), new HingeLoss());
    learner.setRandom(new Random(1337));
    learner.setNumPasses(5);
    return learner;
  }

  public double computeClassificationAccuracy(List<FeatureOutcomePair> data,
      RegressionModel model) {
    return computeClassificationAccuracy(data, model, (v) -> v);
  }

  public double computeClassificationAccuracy(List<FeatureOutcomePair> data,
      RegressionModel model,
      Function<DoubleVector, DoubleVector> clippingFunction) {

    double correct = 0;
    RegressionClassifier clf = new RegressionClassifier(model);
    for (FeatureOutcomePair pair : data) {
      DoubleVector prediction = clippingFunction.apply(clf.predict(pair
          .getFeature()));
      if (prediction.subtract(pair.getOutcome()).abs().sum() < 0.1d) {
        correct++;
      }
    }
    return correct / data.size();
  }

  public List<FeatureOutcomePair> generateDataAddedNoise(int noiseFeatures) {
    return generateData()
        .stream()
        .map(
            (featOut) -> {
              DoubleVector oldFeature = featOut.getFeature();
              DoubleVector feat = new DenseDoubleVector(oldFeature
                  .getDimension() + noiseFeatures);

              for (int i = 0; i < oldFeature.getDimension(); i++) {
                feat.set(i, oldFeature.get(i));
              }

              Random random = new Random(1337);
              for (int i = 0; i < noiseFeatures; i++) {
                feat.set(i + oldFeature.getDimension(), random.nextDouble());
              }

              return new FeatureOutcomePair(feat, featOut.getOutcome());
            }).collect(Collectors.toList());
  }

  public List<FeatureOutcomePair> generateData() {
    return generateData(0);
  }

  public List<FeatureOutcomePair> generateData(double negativeClasValue) {
    return IntStream
        .range(1, 2000)
        .mapToObj(
            (i) -> {
              double mean = i % 2 == 0 ? 25d : 75d;
              double stddev = 10d;
              double clzVal = i % 2 == 0 ? negativeClasValue : 1d;
              double[] feat = new double[] { 1, rnd.nextGaussian(mean, stddev),
                  rnd.nextGaussian(mean, stddev) };

              return new FeatureOutcomePair(new DenseDoubleVector(feat),
                  new SingleEntryDoubleVector(clzVal));
            }).collect(Collectors.toList());
  }
}
