package pll
import breeze.linalg._

case class LeakyReLU(alpha: Double = 0.02) extends Layer {
  var mask: Option[List[DenseVector[Double]]] = None

  def lrelu(x:Double) = {
    if(x >= 0) { x } else { alpha * x }
  }

  def forward(x: DenseVector[Double]) = {
    val out = x.map(lrelu)
    this.mask = Some(out.map(i => if(i >= 0d) 1d else alpha) :: this.mask.getOrElse(Nil))
    out
  }

  def backward(d: DenseVector[Double]) = {
    val m = this.mask.get.head
    this.mask = Some(this.mask.get.tail)
    m *:* d
  }

  def update() {}
  def reset() {
    this.mask = None
  }
  def save(filename: String) {}
  def load(filename: String) {}
  def load(data: List[String]) = { data }
  override def duplicate()={
    val dup = new LeakyReLU(alpha)
    dup.mask = this.mask
    dup
  }
}
