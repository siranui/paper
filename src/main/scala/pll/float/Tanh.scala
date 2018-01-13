package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

class Tanh() extends Layer {
  var y: DenseVector[T] = null
  def forward(x: DenseVector[T]): DenseVector[T] = {
    y = x.map(a => (math.exp(a) - math.exp(-a)) / (math.exp(a) + math.exp(-a)))
    y
  }

  def backward(d: DenseVector[T]): DenseVector[T] = {
    val d1 = d *:* ((1: T) - y *:* y)
    d1
  }
  def update() {}
  def reset() {}
  def save(fn: String) {}
  override def save_(pw: java.io.PrintWriter): java.io.PrintWriter = {
    /* do nothing */
    pw
  }
  def load(fn: String) {}
  override def load(data: List[String]) = {
    data
  }
  override def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {
    /* do nothing */
  }

  override def duplicate() = {
    new Tanh()
  }
}
