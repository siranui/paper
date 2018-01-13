package pll.float

import breeze.linalg._
import typeAlias._

case class ReLU() extends Layer {

  var x1: List[DenseVector[T]] = Nil
  def forward(x: DenseVector[T]): DenseVector[T] = {
    x1 = x.copy :: x1
    val y = new DenseVector[T](x.size)
    for (i <- 0 until x.size) {
      if (x(i) < 0) y(i) = 0
      else y(i) = x(i)
    }
    y
  }

  def backward(d: DenseVector[T]): DenseVector[T] = {
    assert(x1 != null)
    val x = x1.head
    x1 = x1.tail
    for (i <- 0 until d.size) {
      if (x(i) < 0) d(i) = 0
    }
    d
  }
  def update() {}
  def reset() {
    x1 = Nil
  }

  def save(fn: String) {
    /* do nothing */
  }
  override def save_(pw: java.io.PrintWriter): java.io.PrintWriter = {
    /* do nothing */
    pw
  }
  def load(fn: String) {
    /* do nothing */
  }
  override def load(data: List[String]) = {
    data
  }
  override def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {
    /* do nothing */
  }

  override def duplicate() = {
    val r = new ReLU()
    r.x1 = x1
    r
  }
}
