package de.jungblut.online.minimizer;

import java.util.function.Supplier;
import java.util.stream.Stream;

import de.jungblut.math.DoubleVector;
import de.jungblut.online.ml.FeatureOutcomePair;

public interface StochasticMinimizer {

  /**
   * Minimizes the given stochastic cost function on the supplied streams for
   * the given amount of passes over the data.
   * 
   * @param start the start parameters.
   * @param streamSupplier the supplier for the data training streams.
   * @param costFunction the cost function to minimize.
   * @param numPasses the number of passes over the streams.
   * @param verbose true if progress should be printed to the log.
   * @return the optimized set of parameters.
   */
  public DoubleVector minimize(DoubleVector start,
      Supplier<Stream<FeatureOutcomePair>> streamSupplier,
      StochasticCostFunction costFunction, int numPasses, boolean verbose);

}
