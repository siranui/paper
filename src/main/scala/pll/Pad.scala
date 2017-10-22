package pll


import breeze.linalg._


/*
 *  param
 *  ------
 *  width : width of padding (i.e. width = 1)
 *  ud : "up"conv or "down"conv
 *
 */
case class Pad(channel: Int, width: Int, ud: String = "down") extends Layer {

  var xs: List[DenseVector[Double]] = List[DenseVector[Double]]()

  def forward(x: DenseVector[Double]): DenseVector[Double] = {
    xs = x :: xs
    val xch = utils.divideIntoN(x, channel)
    val padded_xch = xch.map(padding(_, width, ud))
    padded_xch.reduceLeft((i, j) => DenseVector.vertcat(i, j)) // convert to 1d
  }

  def backward(d: DenseVector[Double]): DenseVector[Double] = {
    val dch = utils.divideIntoN(d, channel)
    val unpadded_dch = dch.map(back(_, width, ud))
    unpadded_dch.reduceLeft((i, j) => DenseVector.vertcat(i, j)) // convert to 1d
  }

  def update() {}

  def reset() {
    xs = Nil
  }

  def save(filename: String) {}

  def load(filename: String) {}

  def load(data: List[String]): List[String] = {
    data
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
  def padding(m: DenseVector[Double], width: Int, method: String): DenseVector[Double] = {
    val w = math.sqrt(m.size).toInt
    val mat = reshape(m, w, w)

    method match {
      case "up" | "Up" | "UP" => trans(mat.t, width).t.toDenseVector
      case "down" | "Down" | "DOWN" | _ => around(mat.t, width).t.toDenseVector
    }
  }

  def around(m: DenseMatrix[Double], width: Int): DenseMatrix[Double] = {
    val pl = padLeft(m, (m.rows + width, m.cols + width), 0d)
    padRight(pl, (pl.rows + width, pl.cols + width), 0d)
  }

  def trans(m: DenseMatrix[Double], width: Int): DenseMatrix[Double] = {
    assert(m.rows == m.cols, "Rectangle is not support.")

    val w = m.rows
    val ww = (width + 1) * w + width

    val builder = new CSCMatrix.Builder[Double](rows = ww, cols = ww)

    for (i <- width until ww by width + 1; j <- width until ww by width + 1) {
      builder.add(i, j, m(i / (width + 1), j / (width + 1)))
    }
    builder.result.toDense
  }

  def back(d: DenseVector[Double], width: Int, method: String): DenseVector[Double] = {
    import math.sqrt
    val dmat = reshape(d, sqrt(d.size).toInt, sqrt(d.size).toInt).t

    val unpadded_dmat: DenseMatrix[Double] = method match {
      case "up" | "Up" | "UP" =>
        val mat = DenseMatrix.zeros[Double](
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

