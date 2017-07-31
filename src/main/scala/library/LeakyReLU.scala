package pll
import breeze.linalg._

case class LeakyReLU(alpha: Double = 0.02) extends Layer {
  var mask: Option[DenseVector[Double]] = None

  def lrelu(x:Double) = {
    if(x >= 0) { x } else { alpha * x }
  }

  def forward(x: DenseVector[Double]) = {
    val out = x.map(lrelu)
    this.mask = Some(out.map(i => if(i >= 0d) 1d else alpha))
    out
  }

  def backward(d: DenseVector[Double]) = {
    this.mask.get *:* d
  }

  def update() {}
  def reset() {
    this.mask = None
  }
  def save(filename: String) {}
  def load(data: List[String]) = { data }
  override def duplicate()={
    new LeakyReLU(alpha)
  }
}
