package de.jungblut.online.minimizer;

import static org.junit.Assert.assertEquals;

import java.util.function.Supplier;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.Test;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.dense.DenseDoubleVector;
import de.jungblut.math.dense.SingleEntryDoubleVector;
import de.jungblut.math.minimize.CostGradientTuple;
import de.jungblut.online.minimizer.StochasticGradientDescent.StochasticGradientDescentBuilder;
import de.jungblut.online.ml.FeatureOutcomePair;

public class TestStochasticGradientDescent {

  @Test
  public void testGradientDescent() {

    DoubleVector start = new DenseDoubleVector(new double[] { 2, -1 });

    StochasticCostFunction inlineFunction = getCostFunction();

    DoubleVector minimizeFunction = StochasticGradientDescentBuilder
        .create(0.5d).build()
        .minimize(start, fakeStream(), inlineFunction, 10, false);

    // 1E-5 is close enough to zero for the test to pass
    assertEquals(minimizeFunction.get(0), 0, 1E-5);
    assertEquals(minimizeFunction.get(1), 0, 1E-5);
  }

  @Test
  public void testMomentumGradientDescent() {

    DoubleVector start = new DenseDoubleVector(new double[] { 2, -1 });

    StochasticCostFunction inlineFunction = getCostFunction();
    StochasticGradientDescent gd = StochasticGradientDescentBuilder
        .create(0.01d).momentum(0.9d).breakOnDifference(1e-20).build();
    DoubleVector minimizeFunction = gd.minimize(start, fakeStream(),
        inlineFunction, 10, false);
    // 1E-5 is close enough to zero for the test to pass
    assertEquals(minimizeFunction.get(0), 0, 1E-5);
    assertEquals(minimizeFunction.get(1), 0, 1E-5);
  }

  StochasticCostFunction getCostFunction() {
    // our function is f(x,y) = x^2+y^2
    // the derivative is f'(x,y) = 2x+2y
    StochasticCostFunction inlineFunction = new StochasticCostFunction() {

      @Override
      public CostGradientTuple observeExample(FeatureOutcomePair next,
          DoubleVector input) {
        double cost = Math.pow(input.get(0), 2) + Math.pow(input.get(1), 2);
        DenseDoubleVector gradient = new DenseDoubleVector(new double[] {
            input.get(0) * 2, input.get(1) * 2 });

        return new CostGradientTuple(cost, gradient);
      }
    };
    return inlineFunction;
  }

  private Supplier<Stream<FeatureOutcomePair>> fakeStream() {
    return () -> (IntStream.range(0, 100)
        .mapToObj((i) -> new FeatureOutcomePair(new SingleEntryDoubleVector(i),
            new SingleEntryDoubleVector(i))));
  }

}
