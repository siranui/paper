package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

class SoftMax() extends Layer {
  def forward(x: DenseVector[T]) = {
    var esum: T = (0: T)
    val y       = DenseVector.zeros[T](x.size)
    // Max-Value is subtracted from numerator and denominator
    // so that the value is too large to overflow.
    var xmax = x(0)

    for (i <- 1 until x.size) {
      if (x(i) > xmax)
        xmax = x(i)
    }

    for (i <- 0 until x.size) {
      esum += (math.exp(x(i) - xmax): T)
    }

    for (i <- 0 until x.size) {
      y(i) = (math.exp(x(i) - xmax): T) / esum
    }

    y
  }
  def backward(d: DenseVector[T]) = {
    //println(d)
    d
  }
  def update() {}
  def reset() {}
  def save(fn: String) {}
  def load(fn: String) {}
  override def load(data: List[String]) = {
    data
  }
  override def duplicate() = {
    new SoftMax()
  }
}
