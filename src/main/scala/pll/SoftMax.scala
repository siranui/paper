package pll
import breeze.linalg._

class SoftMax() extends Layer {
  def forward(x: DenseVector[Double]) = {
    var esum = 0.0
    val y    = DenseVector.zeros[Double](x.size)
    // Max-Value is subtracted from numerator and denominator
    // so that the value is too large to overflow.
    var xmax = x(0)

    for (i <- 1 until x.size) {
      if (x(i) > xmax)
        xmax = x(i)
    }

    for (i <- 0 until x.size) {
      esum += math.exp(x(i) - xmax)
    }

    for (i <- 0 until x.size) {
      y(i) = math.exp(x(i) - xmax) / esum
    }

    y
  }
  def backward(d: DenseVector[Double]) = {
    //println(d)
    d
  }
  def update() {}
  def reset() {}
  def save(fn: String) {}
  def load(fn: String) {}
  def load(data: List[String]) = {
    data
  }
  override def duplicate() = {
    new SoftMax()
  }
}
