package pll.float

import breeze.linalg._
import typeAlias._
// import CastImplicits._

/*
 *  param
 *  ------
 *  width : width of padding (i.e. width = 1)
 *  ud : "up"conv or "down"conv
 *
 */
case class Pad(channel: Int, width: Int, ud: String = "down") extends Layer {

  var xs: List[DenseVector[T]] = List[DenseVector[T]]()

  def forward(x: DenseVector[T]): DenseVector[T] = {
    xs = x :: xs
    val xch        = utils.divideIntoN(x, channel)
    val padded_xch = xch.map(padding(_, width, ud))
    padded_xch.reduceLeft((i, j) => DenseVector.vertcat(i, j)) // convert to 1d
  }

  def backward(d: DenseVector[T]): DenseVector[T] = {
    val dch          = utils.divideIntoN(d, channel)
    val unpadded_dch = dch.map(back(_, width, ud))
    unpadded_dch.reduceLeft((i, j) => DenseVector.vertcat(i, j)) // convert to 1d
  }

  def update() {}

  def reset() {
    xs = Nil
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

  override def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {
    /* do nothing */
  }

  /*
   *  param
   *  ------
   *  m : Matrix to be padded.
   *  width : width of padding. (i.e. width = 1)
   *
   *  return
   *  ------
   *  padded matrix
   */
  def padding(m: DenseVector[T], width: Int, method: String): DenseVector[T] = {
    val w   = math.sqrt(m.size).toInt
    val mat = reshape(m, w, w)

    method match {
      case "up" | "Up" | "UP"           => trans(mat.t, width).t.toDenseVector
      case "down" | "Down" | "DOWN" | _ => around(mat.t, width).t.toDenseVector
    }
  }

  def around(m: DenseMatrix[T], width: Int): DenseMatrix[T] = {
    val pl = padLeft(m, (m.rows + width, m.cols + width), (0: T))
    padRight(pl, (pl.rows + width, pl.cols + width), (0: T))
  }

  def trans(m: DenseMatrix[T], width: Int): DenseMatrix[T] = {
    assert(m.rows == m.cols, "Rectangle is not support.")

    val w  = m.rows
    val ww = (width + 1) * w + width

    val builder = new CSCMatrix.Builder[T](rows = ww, cols = ww)

    for (i <- width until ww by width + 1; j <- width until ww by width + 1) {
      builder.add(i, j, m(i / (width + 1), j / (width + 1)))
    }
    builder.result.toDense
  }

  def back(d: DenseVector[T], width: Int, method: String): DenseVector[T] = {
    import math.sqrt
    val dmat = reshape(d, sqrt(d.size).toInt, sqrt(d.size).toInt).t

    val unpadded_dmat: DenseMatrix[T] = method match {
      case "up" | "Up" | "UP" =>
        val mat = DenseMatrix.zeros[T](
          (dmat.rows - width) / (width + 1),
          (dmat.cols - width) / (width + 1)
        )

        for {
          i <- width until dmat.rows - width by width + 1
          j <- width until dmat.cols - width by width + 1
        } {
          mat(i / (width + 1), j / (width + 1)) = dmat(i, j)
        }
        mat
      case "down" | "Down" | "DOWN" | _ =>
        dmat(width until dmat.rows - width, width until dmat.cols - width)
    }
    unpadded_dmat.t.toDenseVector
  }

}
