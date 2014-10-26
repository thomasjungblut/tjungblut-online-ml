package de.jungblut.online.regression;

import com.google.common.base.Preconditions;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.activation.ActivationFunction;
import de.jungblut.math.dense.SingleEntryDoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.math.squashing.ErrorFunction;
import de.jungblut.online.minimizer.StochasticMinimizer;
import de.jungblut.online.ml.AbstractMinimizingOnlineLearner;
import de.jungblut.online.ml.FeatureOutcomePair;

/**
 * A regression learner that learns weights on a stream, given an optimization
 * objective (e.g. log loss). This learner outputs a RegressionModel that can be
 * used in a RegressionClassifier.
 * 
 * @author thomas.jungblut
 *
 */
public class RegressionLearner extends
    AbstractMinimizingOnlineLearner<RegressionModel> {

  private final ActivationFunction activationFunction;
  private final ErrorFunction lossFunction;

  public RegressionLearner(StochasticMinimizer minimizer,
      ActivationFunction activationFunction, ErrorFunction lossFunction) {
    super(minimizer);
    this.activationFunction = Preconditions.checkNotNull(activationFunction,
        "activation function");
    this.lossFunction = Preconditions.checkNotNull(lossFunction,
        "loss function");
  }

  @Override
  protected CostGradientTuple observeExample(FeatureOutcomePair next,
      DoubleVector weights) {
    DoubleVector hypothesis = new SingleEntryDoubleVector(
        activationFunction.apply(next.getFeature().dot(weights)));
    double cost = lossFunction.calculateError(next.getOutcome(), hypothesis);
    double diff = hypothesis.subtract(next.getOutcome()).sum();
    // TODO if we want to support other derivations of the gradient, we need to
    // put them into the interface
    DoubleVector gradient = next.getFeature().multiply(diff);
    return new CostGradientTuple(cost, gradient);
  }

  @Override
  protected RegressionModel createModel(DoubleVector weights) {
    return new RegressionModel(weights, activationFunction);
  }
}
