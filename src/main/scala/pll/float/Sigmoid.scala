package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

class Sigmoid() extends Layer {
  var yl = List[DenseVector[T]]()
  //var y:DenseVector[T] = null
  def forward(x: DenseVector[T]): DenseVector[T] = {
    val y = x.map(a => 1 / (1 + math.exp(-a)))
    yl = y :: yl
    y
  }
  def backward(d: DenseVector[T]): DenseVector[T] = {
    // y *:* (1d - y)
    val dd = yl(0) *:* d *:* ((1: T) - yl(0))
    yl = yl.tail
    dd
  }
  def update() {}
  def reset() {
    yl = List[DenseVector[T]]()
  }
  def save(fn: String) {}
  def load(fn: String) {}
  override def load(data: List[String]) = {
    data
  }
  override def duplicate() = {
    new Sigmoid()
  }
}
