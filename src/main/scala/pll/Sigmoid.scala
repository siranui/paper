package pll
import breeze.linalg._

class Sigmoid() extends Layer {
  var yl = List[DenseVector[Double]]()
  //var y:DenseVector[Double] = null
  def forward(x: DenseVector[Double]) = {
    val y = x.map(a => 1 / (1 + math.exp(-a)))
    yl = y :: yl
    y
  }
  def backward(d: DenseVector[Double]) = {
    // y *:* (1d - y)
    val dd = yl(0) *:* d *:* (1d - yl(0))
    yl = yl.tail
    dd
  }
  def update() {}
  def reset() {
    yl = List[DenseVector[Double]]()
  }
  def save(fn: String) {}
  def load(fn: String) {}
  def load(data: List[String]) = {
    data
  }
  override def duplicate() = {
    new Sigmoid()
  }
}
