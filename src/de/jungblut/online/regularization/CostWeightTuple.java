package de.jungblut.online.regularization;

import de.jungblut.math.DoubleVector;

public class CostWeightTuple {

  private final double cost;
  private final DoubleVector weight;

  public CostWeightTuple(double cost, DoubleVector weight) {
    this.cost = cost;
    this.weight = weight;
  }

  public double getCost() {
    return this.cost;
  }

  public DoubleVector getWeight() {
    return this.weight;
  }

}
