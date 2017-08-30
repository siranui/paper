package pll


import breeze.linalg._

class BatchNorm(update_method: String = "SGD", lr: Double = 0.01) extends Layer {
  // 学習対象
  var gamma: DenseVector[Double] = null //DenseVector.ones[Double](ch*dim)
  var beta: DenseVector[Double] = null //DenseVector.zeros[Double](ch*dim)

  // backward()で使うため、forward()の外で定義している
  var xc = Array[DenseVector[Double]]()
  var std = DenseVector[Double]()
  var xn = Array[DenseVector[Double]]()

  // 更新量を保持する変数
  var dgamma: DenseVector[Double] = null //DenseVector.zeros[Double](ch*dim)
  var dbeta: DenseVector[Double] = null //DenseVector.zeros[Double](ch*dim)

  var opt = Opt.create(update_method, lr)
  // opt.register(Array(gamma,beta))

  val eps = 1e-8

  override def forwards(B: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    if (gamma == null) {
      gamma = DenseVector.ones[Double](B(0).length)
      beta = DenseVector.zeros[Double](B(0).length)
      dgamma = DenseVector.zeros[Double](B(0).length)
      dbeta = DenseVector.zeros[Double](B(0).length)
      opt.register(Array(gamma, beta))
    }

    val batch_size = B.length.toDouble

    val mu = B.reduce(_ + _) /:/ batch_size
    xc = B.map(_ - mu)

    val sig2 = xc.map(i => i *:* i).reduce(_ + _) /:/ batch_size
    std = (sig2 +:+ eps).map(math.sqrt)

    xn = xc.map(_ /:/ std)

    xn.map(x => gamma *:* x + beta)
  }

  override def backwards(d: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    dbeta += d.reduce(_ + _)
    dgamma += (xn zip d).map(i => i._1 *:* i._2).reduce(_ + _)

    val batch_size = d.length.toDouble

    val dxn = d.map(_ *:* gamma)

    val dstd = -(dxn zip xc).map(i => (i._1 *:* i._2) /:/ (std *:* std)).reduce(_ + _)
    val dsig2 = 0.5 *:* dstd /:/ std

    val dxc_1 = dxn.map(_ /:/ std)
    val dxc_2 = xc.map(_ *:* 2d *:* dsig2 /:/ batch_size)
    val dxc = (dxc_1 zip dxc_2).map(i => i._1 + i._2)

    val dmu = dxc.reduce(_ + _)

    dxc.map(i => (i - dmu) / batch_size)
  }

  def update() {
    val grads = opt.update(Array(gamma, beta), Array(dgamma, dbeta))

    gamma = gamma - grads(0)
    beta = beta - grads(1)
  }

  def reset() {
    dgamma = DenseVector.zeros[Double](dgamma.length)
    dbeta = DenseVector.zeros[Double](dbeta.length)
    xc = Array[DenseVector[Double]]()
    std = DenseVector[Double]()
    xn = Array[DenseVector[Double]]()
  }

  def forward(x: DenseVector[Double]): DenseVector[Double] = x

  def backward(d: DenseVector[Double]): DenseVector[Double] = d

  def save(filename: String) {}

  def load(filename: String) {}

  def load(data: List[String]): List[String] = {
    data
  }
}
