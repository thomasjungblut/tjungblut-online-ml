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
  private final double lambda;

  public RegressionLearner(StochasticMinimizer minimizer,
      ActivationFunction activationFunction, ErrorFunction lossFunction,
      double ridge) {
    super(minimizer);
    this.activationFunction = Preconditions.checkNotNull(activationFunction,
        "activation function");
    this.lossFunction = Preconditions.checkNotNull(lossFunction,
        "loss function");
    Preconditions.checkArgument(ridge >= 0, "Given ridge lambda was negative: "
        + ridge);
    this.lambda = ridge;
  }

  public RegressionLearner(StochasticMinimizer minimizer,
      ActivationFunction activationFunction, ErrorFunction lossFunction) {
    this(minimizer, activationFunction, lossFunction, 0d);
  }

  @Override
  protected CostGradientTuple observeExample(FeatureOutcomePair next,
      DoubleVector weights) {
    DoubleVector hypothesis = new SingleEntryDoubleVector(
        activationFunction.apply(next.getFeature().dot(weights)));
    double cost = lossFunction.calculateError(next.getOutcome(), hypothesis);
    double diff = hypothesis.subtract(next.getOutcome()).sum();
    DoubleVector gradient = next.getFeature().multiply(diff);

    if (lambda != 0d) {
      boolean bias = next.getFeature().get(0) == 1d;
      DoubleVector powered = weights.pow(2d);
      DoubleVector regGrad = weights.multiply(lambda);
      if (bias) {
        powered.set(0, 0);
        regGrad.set(0, 0);
      }
      cost += lambda * powered.sum() / 2d;
      gradient = gradient.add(regGrad);
    }

    return new CostGradientTuple(cost, gradient);
  }

  @Override
  protected RegressionModel createModel(DoubleVector weights) {
    return new RegressionModel(weights, activationFunction);
  }
}
