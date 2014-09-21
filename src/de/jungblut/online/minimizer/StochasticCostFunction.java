package de.jungblut.online.minimizer;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.online.ml.FeatureOutcomePair;

public interface StochasticCostFunction {

  /**
   * Observes the next example using the given weights.
   * 
   * @param next the next item on the stream.
   * @param weights the current weights.
   * @return a cost/gradient pair that tells the minimizer where to move next.
   */
  public CostGradientTuple observeExample(FeatureOutcomePair next,
      DoubleVector weights);
}
