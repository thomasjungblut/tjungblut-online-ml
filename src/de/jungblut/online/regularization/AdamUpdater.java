package de.jungblut.online.regularization;

import org.apache.commons.math3.util.FastMath;

import com.google.common.base.Preconditions;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;

/**
 * Adam updater, inspired by nd4j. Whitepaper http://arxiv.org/abs/1412.6980
 *
 */
public class AdamUpdater extends GradientDescentUpdater {

  public static final double MOVING_AVERAGE_DECAY = 0.9;
  public static final double SQUARED_DECAY = 0.999;
  public static final double EPS = 1e-8;

  private final double alpha;
  private final double movingAvgDecay;
  private final double squaredDecay;
  private final double eps;

  private DoubleVector movingAvg;
  private DoubleVector squaredGradient;

  public AdamUpdater(double alpha) {
    this(alpha, MOVING_AVERAGE_DECAY, SQUARED_DECAY);
  }

  public AdamUpdater(double alpha, double movingAvgDecay, double squaredDecay) {
    this(alpha, movingAvgDecay, squaredDecay, EPS);
  }

  public AdamUpdater(double alpha, double movingAvgDecay, double squaredDecay,
      double epsilon) {
    Preconditions.checkArgument(movingAvgDecay >= 0 && movingAvgDecay < 1,
        "movingAvgDecay must be [0, 1)!");
    Preconditions.checkArgument(squaredDecay >= 0 && squaredDecay < 1,
        "squaredDecay must be [0, 1)!");
    this.alpha = alpha;
    this.movingAvgDecay = movingAvgDecay;
    this.squaredDecay = squaredDecay;
    this.eps = epsilon;
  }

  @Override
  public CostGradientTuple updateGradient(DoubleVector theta,
      DoubleVector gradient, double learningRate, long iteration, double cost) {

    if (movingAvg == null) {
      // initialize same types with zeros
      movingAvg = gradient.deepCopy().multiply(0);
      squaredGradient = gradient.deepCopy().multiply(0);
    }

    DoubleVector oneMinusBeta1Grad = gradient.multiply(1d - movingAvgDecay);
    movingAvg = movingAvg.multiply(movingAvgDecay).add(oneMinusBeta1Grad);

    DoubleVector oneMinusBeta2GradSquared = gradient.pow(2d).multiply(
        1 - squaredDecay);
    squaredGradient = squaredGradient.multiply(squaredDecay).add(
        oneMinusBeta2GradSquared);

    double beta1t = FastMath.pow(movingAvgDecay, iteration);
    double beta2t = FastMath.pow(squaredDecay, iteration);

    double alphat = alpha * FastMath.sqrt(1 - beta2t) / (1 - beta1t);

    if (Double.isNaN(alphat) || alphat == 0.0) {
      alphat = EPS;
    }

    DoubleVector sqrtV = squaredGradient.sqrt().add(eps);
    gradient = movingAvg.multiply(alphat).divide(sqrtV);

    return new CostGradientTuple(cost, gradient);
  }

}
