package de.jungblut.online.minimizer;

import de.jungblut.math.DoubleVector;

public interface IterationFinishedCallback {

  /**
   * This callback when a pass over an example in a stream of a minimization
   * objective is finished.
   * 
   * @param pass the number of the current pass.
   * @param iteration the number of the current iteration.
   * @param cost the cost at the current iteration.
   * @param currentWeights the current optimal weights.
   * @param validation true if this iteration was used for validation.
   */
  public void onIterationFinished(int pass, long iteration, double cost,
      DoubleVector currentWeights, boolean validation);

}
