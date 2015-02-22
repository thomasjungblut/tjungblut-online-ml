package de.jungblut.online.regression.multinomial;

import java.util.function.IntFunction;
import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.common.base.Preconditions;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.activation.ActivationFunction;
import de.jungblut.math.dense.SingleEntryDoubleVector;
import de.jungblut.math.squashing.ErrorFunction;
import de.jungblut.online.minimizer.StochasticMinimizer;
import de.jungblut.online.ml.AbstractOnlineLearner;
import de.jungblut.online.ml.FeatureOutcomePair;
import de.jungblut.online.regression.RegressionLearner;
import de.jungblut.online.regression.RegressionModel;

/**
 * A regression learner that learns multiple independent regression models and
 * blends them into a single model.
 * 
 * @author thomas.jungblut
 *
 */
public class MultinomialRegressionLearner extends
    AbstractOnlineLearner<MultinomialRegressionModel> {

  private static final Logger LOG = LogManager
      .getLogger(MultinomialRegressionLearner.class);

  private static final SingleEntryDoubleVector POSITIVE = new SingleEntryDoubleVector(
      1d);
  private static final SingleEntryDoubleVector NEGATIVE = new SingleEntryDoubleVector(
      0d);

  private final IntFunction<RegressionLearner> learnerFactory;

  private RegressionModel[] trainedModels;

  public MultinomialRegressionLearner(StochasticMinimizer minimizer,
      ActivationFunction activationFunction, ErrorFunction lossFunction) {
    this((i) -> new RegressionLearner(minimizer, activationFunction,
        lossFunction));
  }

  public MultinomialRegressionLearner(
      IntFunction<RegressionLearner> learnerFactory) {
    this.learnerFactory = Preconditions.checkNotNull(learnerFactory,
        "learnerFactory");
  }

  @Override
  public MultinomialRegressionModel train(
      Supplier<Stream<FeatureOutcomePair>> streamSupplier) {

    init(streamSupplier);

    trainedModels = new RegressionModel[numOutcomeClasses];

    // train the models in parallel
    IntStream
        .range(0, numOutcomeClasses)
        .parallel()
        .forEach(
            i -> {
              if (verbose) {
                LOG.info("Training class " + i);
              }

              RegressionLearner learner = learnerFactory.apply(i);

              final int k = i;
              trainedModels[i] = learner.train(() -> streamSupplier.get().map(
                  (pair) -> makeBinary(pair, k)));

              if (verbose) {
                LOG.info("Done training class " + i);
              }
            });

    return new MultinomialRegressionModel(trainedModels);
  }

  private static FeatureOutcomePair makeBinary(FeatureOutcomePair input,
      int targetClassIndex) {
    DoubleVector outcome = input.getOutcome();
    if (outcome.maxIndex() == targetClassIndex) {
      return new FeatureOutcomePair(input.getFeature(), POSITIVE);
    }

    return new FeatureOutcomePair(input.getFeature(), NEGATIVE);
  }
}
