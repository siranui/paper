package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

case class LeakyReLU(alpha: T = 0.02) extends Layer {
  var mask: Option[List[DenseVector[T]]] = None

  def lrelu(x: T): T = {
    if (x >= 0) x else alpha * x
  }

  def forward(x: DenseVector[T]): DenseVector[T] = {
    val out = x.map(lrelu)
    this.mask = Some(out.map(i => if (i >= (0: T)) (1: T) else alpha) :: this.mask.getOrElse(Nil))
    out
  }

  def backward(d: DenseVector[T]): DenseVector[T] = {
    val m = this.mask.get.head
    this.mask = Some(this.mask.get.tail)
    m *:* d
  }

  def update() {}

  def reset() {
    this.mask = None
  }

  def save(filename: String) {}

  override def save_(pw: java.io.PrintWriter): java.io.PrintWriter = {
    /* do nothing */
    pw
  }

  def load(filename: String) {}

  override def load(data: List[String]): List[String] = {
    data
  }

  override def duplicate() = {
    val dup = new LeakyReLU(alpha)
    dup.mask = this.mask
    dup
  }
}