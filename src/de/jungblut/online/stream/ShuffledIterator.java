package de.jungblut.online.stream;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

import com.google.common.base.Preconditions;
import com.google.common.collect.AbstractIterator;

/**
 * A shuffled iterator. The implementation buffers a fixed amount of data by
 * consuming the stream, shuffles it and makes it available as a stream again.
 */
public final class ShuffledIterator<T> extends AbstractIterator<T> {

  private final Iterator<T> baseStreamIterator;
  private final int bufferedItems;
  private final List<T> buffer;
  private int currentIndex;

  ShuffledIterator(Stream<T> baseStream, int bufferedItems) {
    Preconditions.checkState(bufferedItems > 0, "bufferedItems > 0");
    Preconditions.checkNotNull(baseStream, "baseStream");

    this.bufferedItems = bufferedItems;
    this.baseStreamIterator = baseStream.iterator();
    this.buffer = new ArrayList<>(bufferedItems);

    bufferAndShuffle();
  }

  @Override
  protected T computeNext() {
    if (currentIndex >= buffer.size()) {
      bufferAndShuffle();
      if (buffer.isEmpty()) {
        return endOfData();
      }
    }

    return buffer.get(currentIndex++);
  }

  public Stream<T> asStream() {
    return StreamSupport.stream(
        Spliterators.spliteratorUnknownSize(this, Spliterator.NONNULL), false);
  }

  private void bufferAndShuffle() {
    buffer.clear();
    currentIndex = 0;
    for (int i = 0; i < bufferedItems; i++) {
      if (baseStreamIterator.hasNext()) {
        buffer.add(baseStreamIterator.next());
      } else {
        break;
      }
    }
    Collections.shuffle(buffer);
  }

  /**
   * Creates a new shuffled iterator to "proxy shuffle" a stream with the given
   * shuffle buffer.
   * 
   * @param baseStream the base stream to load elements from.
   * @param bufferedItems the buffer size used to shuffle items.
   * @return a shuffled stream.
   */
  public static <T> Stream<T> fromStream(Stream<T> baseStream, int bufferedItems) {
    return new ShuffledIterator<>(baseStream, bufferedItems).asStream();
  }

}
