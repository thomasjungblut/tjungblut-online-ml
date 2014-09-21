package de.jungblut.online.ml;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.mockito.Mockito;

import de.jungblut.math.dense.DenseDoubleVector;

public class TestAbstractMinimizingOnlineLearner {

  @Test
  public void testPeekDimensions() {
    // TODO please someone implement a parameterized junit test that doesn't
    // suck..

    assertDimensionsMatch(5, 2, 2);
    assertDimensionsMatch(5, 1, 2);
    assertDimensionsMatch(5, 15, 15);
    assertDimensionsMatch(25, 1, 2);
  }

  @Test(expected = NullPointerException.class)
  public void testNullStream() {
    AbstractMinimizingOnlineLearner<?> mock = Mockito.mock(
        AbstractMinimizingOnlineLearner.class, Mockito.CALLS_REAL_METHODS);
    mock.peekDimensions(() -> null);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testEmptyStream() {
    AbstractMinimizingOnlineLearner<?> mock = Mockito.mock(
        AbstractMinimizingOnlineLearner.class, Mockito.CALLS_REAL_METHODS);
    mock.peekDimensions(() -> new ArrayList<FeatureOutcomePair>().stream());
  }

  private void assertDimensionsMatch(int features, int outcome, int classes) {
    AbstractMinimizingOnlineLearner<?> mock = Mockito.mock(
        AbstractMinimizingOnlineLearner.class, Mockito.CALLS_REAL_METHODS);
    List<FeatureOutcomePair> list = new ArrayList<>();
    list.add(new FeatureOutcomePair(new DenseDoubleVector(features),
        new DenseDoubleVector(outcome)));

    mock.peekDimensions(() -> list.stream());

    Assert.assertEquals(features, mock.featureDimension);
    Assert.assertEquals(outcome, mock.outcomeDimension);
    Assert.assertEquals(classes, mock.numOutcomeClasses);
  }

}
