package de.jungblut.online.ml;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public interface Model {

  public void serialize(DataOutput out) throws IOException;

  public Model deserialize(DataInput in) throws IOException;

}
