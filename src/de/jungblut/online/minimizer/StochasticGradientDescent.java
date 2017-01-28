package de.jungblut.online.minimizer;

import java.util.Deque;
import java.util.LinkedList;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.StampedLock;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.online.ml.FeatureOutcomePair;
import de.jungblut.online.regularization.CostWeightTuple;
import de.jungblut.online.regularization.GradientDescentUpdater;
import de.jungblut.online.regularization.WeightUpdater;

/**
 * Stochastic gradient descent. This class is designed to work on a parallel
 * stream and do stochastic updates to a parameter set.
 * 
 * @author thomas.jungblut
 *
 */
public class StochasticGradientDescent implements StochasticMinimizer {

  private static final Logger LOG = LogManager
      .getLogger(StochasticGradientDescent.class);

  public static class StochasticGradientDescentBuilder {

    private final double alpha;
    private double breakDifference;
    private double momentum;
    private int historySize = 10;
    private int progressReportInterval = 1;
    private double holdoutValidationPercentage = 0d;
    private boolean adaptiveLearningRate = false;
    private WeightUpdater weightUpdater = new GradientDescentUpdater();
    private long validationRandomSeed = System.currentTimeMillis();

    private StochasticGradientDescentBuilder(double alpha) {
      this.alpha = alpha;
    }

    public StochasticGradientDescent build() {
      return new StochasticGradientDescent(this);
    }

    /**
     * Add momentum to this gradient descent minimizer.
     * 
     * @param momentum the momentum to use. Between 0 and 1.
     * @return the builder again.
     */
    public StochasticGradientDescentBuilder momentum(double momentum) {
      Preconditions.checkArgument(momentum >= 0d && momentum <= 1d,
          "Momentum must be between 0 and 1.");
      this.momentum = momentum;
      return this;
    }

    /**
     * In order to fix the reproducibility of a given train/test set split, you
     * can pass the seed value.
     * 
     * @param seed the seed as passed into {@link java.util.Random}.
     * @return the builder again.
     */
    public StochasticGradientDescentBuilder validationRandomSeed(long seed) {
      this.validationRandomSeed = seed;
      return this;
    }

    /**
     * Holdout validation percentage, this will take a subset of the data on the
     * stream and do a validation on it.
     * 
     * @param perc the percentage to use. Between 0 and 1.
     * @return the builder again.
     */
    public StochasticGradientDescentBuilder holdoutValidationPercentage(
        double perc) {
      Preconditions.checkArgument(momentum >= 0d && momentum <= 1d,
          "HoldOut Percentage must be between 0 and 1.");
      this.holdoutValidationPercentage = perc;
      return this;
    }

    /**
     * Sets the weight updater, for example to use regularization. The default
     * is the normal gradient descent.
     * 
     * To set the regularization parameter use the {@link #lambda(double)}
     * method.
     * 
     * @param weightUpdater the updater to use.
     * @return the builder again.
     */
    public StochasticGradientDescentBuilder weightUpdater(
        WeightUpdater weightUpdater) {
      this.weightUpdater = Preconditions.checkNotNull(weightUpdater);
      return this;
    }

    /**
     * Sets the size of the history to keep to compute average improvements and
     * output progress information.
     * 
     * @return the builder again.
     */
    public StochasticGradientDescentBuilder historySize(int historySize) {
      Preconditions.checkArgument(historySize > 0, "HistorySize must be > 0");
      this.historySize = historySize;
      return this;
    }

    /**
     * Sets the progress report interval. Since writing to the console/log might
     * be expensive, this is an easy way to limit the logging if needed.
     * 
     * @param interval the interval. E.g. every 10th iteration.
     * @return the builder again.
     */
    public StochasticGradientDescentBuilder progressReportInterval(int interval) {
      Preconditions.checkArgument(interval > 0, "ReportInterval must be > 0");
      this.progressReportInterval = interval;
      return this;
    }

    /**
     * Breaks minimization process when the given delta in costs have been
     * archieved. Usually a quite low value of 1e-4 to 1e-8.
     * 
     * @param delta the delta to break in difference between two costs.
     * @return the builder again.
     */
    public StochasticGradientDescentBuilder breakOnDifference(double delta) {
      this.breakDifference = delta;
      return this;
    }

    /**
     * Enables adaptive learning rate, using the algorithm: <br/>
     * 
     * <pre>
     * alpha = 1d / (initialAlpha * (allIterations + 2));
     * </pre>
     * 
     * where allIterations is a counter over all passes.
     * 
     * @return the builder again
     */
    public StochasticGradientDescentBuilder enableAdaptiveLearningRate() {
      this.adaptiveLearningRate = true;
      return this;
    }

    /**
     * Creates a new builder.
     * 
     * @param alpha the learning rate to set.
     * @return a new builder.
     */
    public static StochasticGradientDescentBuilder create(double alpha) {
      return new StochasticGradientDescentBuilder(alpha);
    }

  }

  private final StochasticGradientDescentBuilder builder;
  private final long validationSeed;

  private IterationFinishedCallback iterationCallback;
  private ValidationFinishedCallback validationCallback;
  private PassFinishedCallback passCallback;

  private double breakDifference;
  private double momentum;
  private double initialAlpha;
  private double validationPercentage;
  private int historySize;
  private int progressReportInterval;
  private WeightUpdater weightUpdater;
  private StampedLock lock = new StampedLock();

  // we are fixing the random for validation to generate the same sequences
  // to not mix train and validation set.
  private Random validationRandom;
  private Deque<Double> costHistory;
  private DoubleVector lastTheta = null;
  private DoubleVector theta;
  private double alpha;
  private int validationItems;
  private double validationError;
  private double trainingError;
  private boolean stopAfterThisPass = false;
  private boolean adaptiveLearningRate = false;
  private long iteration = 0;
  private long allIterations = 0;
  private Stopwatch startWatch;

  private StochasticGradientDescent(StochasticGradientDescentBuilder builder) {
    this.builder = builder;
    this.validationSeed = builder.validationRandomSeed;
    resetState(builder);
  }

  private void resetState(StochasticGradientDescentBuilder builder) {
    this.initialAlpha = builder.alpha;
    this.alpha = this.initialAlpha;
    this.breakDifference = builder.breakDifference;
    this.momentum = builder.momentum;
    this.progressReportInterval = builder.progressReportInterval;
    this.historySize = builder.historySize;
    this.weightUpdater = builder.weightUpdater;
    this.validationPercentage = builder.holdoutValidationPercentage;
    this.adaptiveLearningRate = builder.adaptiveLearningRate;
    this.costHistory = new LinkedList<>();
  }

  @Override
  public DoubleVector minimize(DoubleVector start,
      Supplier<Stream<FeatureOutcomePair>> streamSupplier,
      StochasticCostFunction costFunction, int numPasses, boolean verbose) {

    resetState(builder);
    theta = start;

    startWatch = Stopwatch.createStarted();
    for (int pass = 0; pass < numPasses; pass++) {

      validationRandom = new Random(validationSeed);
      iteration = 0;
      trainingError = 0;
      validationError = 0;
      validationItems = 0;

      Stream<FeatureOutcomePair> currentStream = streamSupplier.get();
      final int passFinal = pass;
      if (currentStream.isParallel()) {
        currentStream.forEach((next) -> doStepLocked(passFinal, next,
            costFunction, verbose));
      } else {
        currentStream.forEach((next) -> doStep(passFinal, next, costFunction,
            verbose));
      }

      if (verbose) {
        LOG.info(String
            .format(
                "Pass Summary %d | Iteration %d | Validation Cost: %g | Training Cost: %g | Iterations/s: %g  | Total Time Taken: %s",
                pass,
                iteration,
                validationError / Math.max(validationItems, 1),
                trainingError / Math.max(iteration - validationItems, 1),
                allIterations
                    / (double) Math.max(startWatch.elapsed(TimeUnit.SECONDS), 1),
                startWatch));
      }

      if (passCallback != null) {
        boolean continuePass = passCallback.onPassFinished(pass, iteration,
            validationError, theta);

        // break this pass, because the callback said so
        if (!continuePass) {
          break;
        }

      }

      if (stopAfterThisPass) {
        break;
      }
    }

    return theta;
  }

  // TODO this write lock is huge, can it be broken down more?
  private void doStepLocked(int pass, FeatureOutcomePair next,
      StochasticCostFunction costFunction, boolean verbose) {
    Lock writeLock = lock.asWriteLock();
    try {
      writeLock.lock();
      doStep(pass, next, costFunction, verbose);
    } finally {
      writeLock.unlock();
    }
  }

  private void doStep(int pass, FeatureOutcomePair next,
      StochasticCostFunction costFunction, boolean verbose) {

    DoubleVector iterationLocalTheta = Preconditions.checkNotNull(weightUpdater
        .prePredictionWeightUpdate(next, theta, alpha, allIterations),
        "weight updater #prePredictionWeightUpdate return must be non-null!");

    CostGradientTuple observed = costFunction.observeExample(next,
        iterationLocalTheta);

    if (verbose) {
      double avgImprovement = getAverageImprovement(costHistory);
      if (iteration > 0 && iteration % progressReportInterval == 0) {
        LOG.info(String
            .format(
                "Pass %d | Iteration %d | Validation Cost: %g | Training Cost: %g | Avg Improvement: %g | Iterations/s: %g",
                pass,
                iteration,
                validationError / Math.max(validationItems, 1),
                trainingError / Math.max(iteration - validationItems, 1),
                avgImprovement,
                allIterations
                    / (double) Math.max(startWatch.elapsed(TimeUnit.SECONDS), 1)));
      }
    }

    dropOldValues(costHistory);

    boolean validation = false;
    if (validationPercentage > 0) {
      if (validationRandom.nextDouble() < validationPercentage) {
        validationError += observed.getCost();
        validationItems++;
        // update the history
        costHistory.addLast(validationError / Math.max(validationItems, 1));
        validation = true;

        if (validationCallback != null) {
          validationCallback.onValidationFinished(pass, iteration,
              observed.getCost(), iterationLocalTheta, next);
        }
      }
    } else {
      costHistory.addLast(observed.getCost() / Math.max(iteration, 1));
    }

    if (iterationCallback != null) {
      iterationCallback.onIterationFinished(pass, iteration,
          observed.getCost(), iterationLocalTheta, validation);
    }

    if (validation) {
      // return to not update the parameters when we did a validation step
      return;
    }

    trainingError += observed.getCost();

    CostWeightTuple update = updateWeights(iterationLocalTheta, observed);

    // save our last parameter
    lastTheta = iterationLocalTheta;
    theta = update.getWeight();

    computeMomentum();

    // break if we converged below the limit
    if (converged(costHistory, breakDifference)) {
      stopAfterThisPass = true;
    }

    allIterations++;
    iteration++;

    if (adaptiveLearningRate) {
      alpha = 1d / (initialAlpha * (allIterations + 2));
    }
  }

  public void computeMomentum() {
    // compute momentum
    if (lastTheta != null && momentum != 0d) {
      // we add momentum as the parameter "m" multiplied by the
      // difference of both theta vectors
      theta = theta.add((lastTheta.subtract(theta)).multiply(momentum));
    }
  }

  public CostWeightTuple updateWeights(DoubleVector iterationLocalTheta,
      CostGradientTuple observed) {
    return weightUpdater.computeNewWeights(iterationLocalTheta,
        observed.getGradient(), alpha, allIterations, observed.getCost());
  }

  public void setIterationCallback(IterationFinishedCallback iterationCallback) {
    this.iterationCallback = iterationCallback;
  }

  public void setValidationCallback(
      ValidationFinishedCallback validationCallback) {
    this.validationCallback = validationCallback;
  }

  public void setPassCallback(PassFinishedCallback passCallback) {
    this.passCallback = passCallback;
  }

  // TODO this should use a cyclic buffer instead of a deque
  private void dropOldValues(Deque<Double> lastCosts) {
    while (lastCosts.size() > historySize) {
      lastCosts.pop();
    }
  }

  private boolean converged(Deque<Double> lastCosts, double limit) {
    return Math.abs(getAverageImprovement(lastCosts)) < limit;
  }

  private double getAverageImprovement(Deque<Double> lastCosts) {
    if (lastCosts.size() >= 2) {
      double first = lastCosts.peek();
      double value = lastCosts.peekLast();
      return (value - first) / lastCosts.size();
    }
    return 0d;
  }

}
