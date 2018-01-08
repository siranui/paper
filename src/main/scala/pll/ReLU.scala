package pll
import breeze.linalg._

case class ReLU() extends Layer {

  var x1: List[DenseVector[Double]] = Nil
  def forward(x: DenseVector[Double]) = {
    x1 = x.copy :: x1
    val y = new DenseVector[Double](x.size)
    for (i <- 0 until x.size) {
      if (x(i) < 0d) y(i) = 0d
      else y(i) = x(i)
    }
    y
  }

  def backward(d: DenseVector[Double]) = {
    assert(x1 != null)
    val x = x1.head
    x1 = x1.tail
    for (i <- 0 until d.size) {
      if (x(i) < 0d) d(i) = 0d
    }
    d
  }
  def update() {}
  def reset() {
    x1 = Nil
  }
  def save(fn: String) {}
  def load(fn: String) {}
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
