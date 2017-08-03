package pll
import breeze.linalg._

class BatchNorm(ch:Int = 1, dim: Int, update_method: String = "SGD", lr: Double = 0.01) extends Layer {
  var opt = Opt.create(update_method,lr)

  val eps = 1e-5

  var gamma = DenseVector.ones[Double](ch*dim)
  var beta = DenseVector.zeros[Double](ch*dim)

  var batch_size = 0d
  var xc = Array[DenseVector[Double]]()
  var sig2 = DenseVector[Double]()
  var std = DenseVector[Double]()
  var xn = Array[DenseVector[Double]]()

  var dgamma = DenseVector.zeros[Double](ch*dim)
  var dbeta = DenseVector.zeros[Double](ch*dim)

  opt.register(Array(gamma,beta))

  override def forwards(B: Array[DenseVector[Double]]) = {
    batch_size = B.size
    val mu = B.reduce(_+_) /:/ batch_size
    xc = B.map(_ - mu)
    sig2 = xc.map(i => i *:* i).reduce(_+_) /:/ batch_size
    std = breeze.numerics.log(sig2 +:+ eps)
    xn = xc.map(_ /:/ std)
    xn.map(x => gamma *:* x + beta)
  }

  override def backwards(d: Array[DenseVector[Double]]) = {
    dbeta += d.reduce(_+_)
    dgamma += (xn zip d).map(i => i._1 dot i._2).reduce(_+_)
    val dxn = d.map(_ dot gamma)
    var dxc = dxn.map(_ /:/ std)
    val dstd = - (dxn zip xc).map(i => (i._1 *:* i._2) /:/ (std *:* std)).reduce(_+_)
    val dsig2 = 0.5 *:* dstd /:/ std
    dxc = (dxc zip xc.map(_ *:* (2d / batch_size) *:* dsig2)).map(i => i._1 + i._2)
    val dmu = dxc.reduce(_+_)
    val dx = dxc.map(i => (i - dmu) / batch_size)
    dx
  }

  def update() {
    val grads = opt.update(Array(gamma,beta),Array(dgamma,dbeta))

    gamma = gamma - grads(0)
    beta = beta - grads(1)
  }

  def reset() = {
    dgamma = DenseVector.zeros[Double](ch*dim)
    dbeta = DenseVector.zeros[Double](ch*dim)
    batch_size = 0d
    xc = Array[DenseVector[Double]]()
    sig2 = DenseVector[Double]()
    std = DenseVector[Double]()
    xn = Array[DenseVector[Double]]()
  }

  def forward(x: DenseVector[Double]): DenseVector[Double] = x
  def backward(d: DenseVector[Double]): DenseVector[Double] = d
  def save(filename: String) {}
  def load(filename: String) {}
  def load(data: List[String]) = {data}
}
