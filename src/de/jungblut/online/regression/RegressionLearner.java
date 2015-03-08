package de.jungblut.online.regression;

import com.google.common.base.Preconditions;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.activation.ActivationFunction;
import de.jungblut.math.dense.SingleEntryDoubleVector;
import de.jungblut.math.loss.LossFunction;
import de.jungblut.math.minimize.CostGradientTuple;
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
  private final LossFunction lossFunction;

  public RegressionLearner(StochasticMinimizer minimizer,
      ActivationFunction activationFunction, LossFunction lossFunction) {
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
    double cost = lossFunction.calculateLoss(next.getOutcome(), hypothesis);
    double deriv = lossFunction.calculateDerivative(next.getOutcome(),
        hypothesis).get(0);
    DoubleVector gradient = next.getFeature().multiply(deriv);
    return new CostGradientTuple(cost, gradient);
  }

  @Override
  public RegressionModel createModel(DoubleVector weights) {
    return new RegressionModel(weights, activationFunction);
  }
}
