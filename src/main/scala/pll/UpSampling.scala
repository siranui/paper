package pll

import breeze.linalg._
// import breeze.stats.mean

trait addKeyWord {
  implicit class Debug(str: String) {
    def debug = s"DEBUG: $str"
  }
}

case class UpSampling1D(sz: Int) extends Layer with addKeyWord {

  def forward(x: DenseVector[Double]) = DenseVector.tabulate(x.length * sz) { i =>
    x(i / sz)
  }
  def backward(d: DenseVector[Double]) = DenseVector.tabulate(d.length / sz) { i =>
    sum(d(i * sz until i * sz + 2))
  }

  def reset(): Unit  = {}
  def update(): Unit = {}

  def save(filename: String): Unit                    = {}
  def load(filename: String): Unit                    = {}
  override def load(data: List[String]): List[String] = data
  override def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {
    /* do nothing */
  }

}

case class UpSampling2D(sz: Int)(implicit input_shape: (Int, Int, Int) = (1, 0, 0))
    extends Layer
    with addKeyWord {

  private var ch  = 0
  private var row = 0
  private var col = 0

  def forward(x: DenseVector[Double]): DenseVector[Double] = {
    val (ch, row, col) = input_shape match {
      case (ch, 0, 0) => (ch, math.sqrt(x.length).toInt, math.sqrt(x.length).toInt)
      case (ch, r, v) => (ch, r, v)
    }

    this.ch = ch
    this.row = row
    this.col = col

    val xs: Array[DenseVector[Double]] = utils.divideIntoN(x, ch)
    val upsample_mats: Array[DenseVector[Double]] = xs.map { m =>
      val reshaped_m = reshape(m, row, col).t
      val upsample_mat = DenseMatrix.tabulate(row * sz, col * sz) {
        case (i, j) => reshaped_m(i / sz, j / sz)
      }
      upsample_mat.t.toDenseVector
    }
    // println(s"${upsample_mats.toList}".debug)

    upsample_mats.reduce(DenseVector.vertcat(_, _))
  }

  def backward(d: DenseVector[Double]) = {
    val ds: Array[DenseVector[Double]] = utils.divideIntoN(d, this.ch)
    val unupsample_mats: Array[DenseVector[Double]] = ds.map { m =>
      val reshaped_m = reshape(m, this.row * sz, this.col * sz).t
      val unupsample_mat = DenseMatrix.tabulate(row, col) {
        case (i, j) => sum(reshaped_m(i * sz until i * sz + sz, j * sz until j * sz + sz))
      }
      unupsample_mat.t.toDenseVector
    }

    unupsample_mats.reduce(DenseVector.vertcat(_, _))
  }

  def reset(): Unit  = {}
  def update(): Unit = {}

  def save(filename: String): Unit = {}
  override def save_(pw: java.io.PrintWriter): java.io.PrintWriter = {
    /* do nothing */
    pw
  }
  def load(filename: String): Unit                    = {}
  override def load(data: List[String]): List[String] = data
  override def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {
    /* do nothing */
  }
}
