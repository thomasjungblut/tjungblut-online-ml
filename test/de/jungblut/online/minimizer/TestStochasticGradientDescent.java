package de.jungblut.online.minimizer;

import static org.junit.Assert.assertEquals;

import java.util.Random;
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
    assertEquals(0, minimizeFunction.get(0), 1E-5);
    assertEquals(0, minimizeFunction.get(1), 1E-5);
  }

  @Test
  public void testMomentumGradientDescent() {

    DoubleVector start = new DenseDoubleVector(new double[] { 2, -1 });

    StochasticCostFunction inlineFunction = getCostFunction();
    StochasticGradientDescent gd = StochasticGradientDescentBuilder
        .create(0.01d).momentum(0.9d).build();
    DoubleVector minimizeFunction = gd.minimize(start, fakeStream(),
        inlineFunction, 100, false);
    // 1E-5 is close enough to zero for the test to pass
    assertEquals(0, minimizeFunction.get(0), 1E-5);
    assertEquals(0, minimizeFunction.get(1), 1E-5);
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

  // this should be a "jdk test" that assures the random class computes
  // predictable sequences given a constant seed (eg across java versions).
  @Test
  public void testRandomSequences() {
    double[] expectedSequence = new double[] { 0.6599297847448217,
        0.6892426740281012, 0.8832726771624211, 0.8985075624657751,
        0.1745695418283183, 0.9419556844395134, 0.14807938070711846,
        0.3187595156771933, 0.813012556775115, 0.6367885304489944,
        0.19570252190931603, 0.49526296435984674, 0.07871964753598903,
        0.2369141939417081, 0.9440162379566716, 0.16136608635562766,
        0.5382951834671054, 0.18435352402091132, 0.5636235622760224,
        0.9924774871892053, 0.44130694778942414, 0.6572159774089178,
        0.961775957472761, 0.18975205873477619, 0.6804044832294249,
        0.22284961674060189, 0.8646027292136993, 0.558639292972846,
        0.5601506490584836, 0.711027533307345, 0.2966305283964148,
        0.36589404251263913, 0.16817544597124956, 0.2768059990539451,
        0.5348940997678332, 0.9822823141408105, 0.6237076247980868,
        0.6744466923806598, 0.5495096488327459, 0.9480977200174032,
        0.11623384969516393, 0.22180404027204093, 0.533402465190628,
        0.6347366203918775, 0.44098128931395175, 0.7988360803301529,
        0.3709506848507138, 0.11796336627975978, 0.6881143461770186,
        0.49440471584477985, 0.7986276989321466, 0.22712934896495163,
        0.8489817571312193, 0.1922900816860258, 0.5702953803536409,
        0.9894925736133386, 0.3654455675395042, 0.47035020346266054,
        0.15562152195173196, 0.8499174516419856, 0.06726866272139875,
        0.6760162536451723, 0.37216433597937204, 0.11361710805819336,
        0.5177333026119711, 0.936820479415565, 0.8799876108654319,
        0.03323324730466415, 0.5191076019358404, 0.1664124702016403,
        0.6522405769374773, 0.8414316930349137, 0.41727316822334837,
        0.8345264805193917, 0.3759563706994753, 0.8163243686388096,
        0.8887716817598789, 0.3836212743363546, 0.23439356321851124,
        0.15941131228817362, 0.09372380196895103, 0.6954207598968032,
        0.6755658318555854, 0.06436911416047997, 0.4345192663957267,
        0.9373423833161629, 0.5811745075973134, 0.859894954127159,
        0.6728719792462856, 0.4518072838793753, 0.7159795534739218,
        0.5209485333873985, 0.8267793988376785, 0.09068481741820189,
        0.45971480935077946, 0.16011642629458878, 0.0941210163767704,
        0.5426311865586962, 0.6019731790639526, 0.3344566363092968 };

    final long seed = 1337;
    Random r = new Random(seed);
    for (int i = 0; i < expectedSequence.length; i++) {
      assertEquals(expectedSequence[i], r.nextDouble(), 1e-4);
    }

  }

  private Supplier<Stream<FeatureOutcomePair>> fakeStream() {
    return () -> (IntStream.range(0, 100)
        .mapToObj((i) -> new FeatureOutcomePair(new SingleEntryDoubleVector(i),
            new SingleEntryDoubleVector(i))));
  }

}
