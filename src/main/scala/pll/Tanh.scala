package pll
import breeze.linalg._

class Tanh() extends Layer {
  var y: DenseVector[Double] = null
  def forward(x: DenseVector[Double]) = {
    y = x.map(a => (math.exp(a) - math.exp(-a)) / (math.exp(a) + math.exp(-a)))
    y
  }

  def backward(d: DenseVector[Double]) = {
    val d1 = d *:* (1d - y *:* y)
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
