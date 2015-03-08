package de.jungblut.classification.eval;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import de.jungblut.math.DoubleVector;
import de.jungblut.online.minimizer.IterationFinishedCallback;
import de.jungblut.online.minimizer.PassFinishedCallback;

public class ErrorCountingCallback implements IterationFinishedCallback,
    PassFinishedCallback {

  private static final Logger LOG = LogManager
      .getLogger(ErrorCountingCallback.class);

  private long errors;
  private long seen;

  @Override
  public void onIterationFinished(int pass, long iteration, double cost,
      DoubleVector currentWeights, boolean validation) {

    if (cost != 0d) {
      errors++;
    }
    seen++;
  }

  @Override
  public boolean onPassFinished(int pass, long iteration, double cost,
      DoubleVector currentWeights) {

    LOG.info("Errors | Pass " + pass + " | Iteration " + iteration
        + " | #Errors " + errors + " | Accuracy " + (errors / (double) seen));

    boolean continueComputation = errors != 0;
    errors = 0; // reset the errors
    seen = 0;
    return continueComputation;
  }
}
