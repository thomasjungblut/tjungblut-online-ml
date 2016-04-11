package de.jungblut.online.stream;

import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.junit.Assert;
import org.junit.Test;

public class TestShuffledIterator {

  @Test
  public void testLessItemsThanBufferSpace() {
    ShuffledIterator<Integer> it = new ShuffledIterator<>(streamHundredItems(),
        1000);
    assertStreamContainsAllItems(it.asStream());
  }

  @Test
  public void testMoreItemsThanBufferSpace() {
    ShuffledIterator<Integer> it = new ShuffledIterator<>(streamHundredItems(),
        10);
    assertStreamContainsAllItems(it.asStream());
  }

  // try to fuzz some off-by-one errors

  @Test
  public void testFuzzyInputEdgeConditions() {
    for (int i = 96; i < 105; i++) {
      ShuffledIterator<Integer> it = new ShuffledIterator<>(
          streamHundredItems(), i);
      assertStreamContainsAllItems(it.asStream());
    }
  }

  public void assertStreamContainsAllItems(Stream<Integer> stream) {
    List<Integer> collected = stream.collect(Collectors.toList());

    // assert uniqueness and integrity
    Assert.assertEquals(100, collected.size());
    HashSet<Integer> set = new HashSet<>(collected);
    Assert.assertEquals(100, set.size());

    for (int i = 0; i < 100; i++) {
      Assert.assertTrue("set didn't contain " + i, set.contains(i));
    }

  }

  public Stream<Integer> streamHundredItems() {
    return IntStream.range(0, 100).boxed();
  }

}
