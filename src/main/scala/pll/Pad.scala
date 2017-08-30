package pll


import breeze.linalg._


/*
 *  param
 *  ------
 *  width : パディングする幅 (i.e. width = 1)
 *  ud : "up"conv or "down"conv
 *
 */
case class Pad(channel: Int, width: Int, ud: String = "down") extends Layer {

  var xs: List[DenseVector[Double]] = List[DenseVector[Double]]()

  def forward(x: DenseVector[Double]): DenseVector[Double] = {
    xs = x :: xs
    val xch = divideIntoN(x, channel)
    val padded_xch = xch.map(padding(_, width, ud))
    padded_xch.reduceLeft((i, j) => DenseVector.vertcat(i, j)) // convert to 1d
  }

  def backward(d: DenseVector[Double]): DenseVector[Double] = {
    val dch = divideIntoN(d, channel)
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
   *  m : パディング対象の行列
   *  width : パディングする幅 (i.e. width = 1)
   *
   *  return
   *  ------
   *  パディングされた行列
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

  def divideIntoN(x: DenseVector[Double], N: Int): Array[DenseVector[Double]] = {
    val len = x.size / N
    (for (i <- 0 until N) yield {
      x(i * len until (i + 1) * len)
    }).toArray
  }
}

// {{{ PadTest
object PadTest {
  def main(args: Array[String]) {

    val v = DenseVector[Double](
      1, 2, 3,
      4, 5, 6,
      7, 8, 9
    )
    val p110d = Pad(channel = 1, width = 1)
    val around0 = DenseVector[Double](
      0, 0, 0, 0, 0,
      0, 1, 2, 3, 0,
      0, 4, 5, 6, 0,
      0, 7, 8, 9, 0,
      0, 0, 0, 0, 0
    )
    assert(p110d.forward(v) == around0, "around0")
    assert(p110d.backward(p110d.forward(v)) == v, "around0 back")
    p110d.reset()


    val v2 = DenseVector[Double](
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1, 2, 3,
      4, 5, 6,
      7, 8, 9
    )
    val p210d = Pad(channel = 2, width = 1)
    val around1 = DenseVector[Double](
      0, 0, 0, 0, 0,
      0, 1, 2, 3, 0,
      0, 4, 5, 6, 0,
      0, 7, 8, 9, 0,
      0, 0, 0, 0, 0,

      0, 0, 0, 0, 0,
      0, 1, 2, 3, 0,
      0, 4, 5, 6, 0,
      0, 7, 8, 9, 0,
      0, 0, 0, 0, 0
    )
    assert(p210d.forward(v2) == around1, "around1")
    assert(p210d.backward(p210d.forward(v2)) == v2, "around1 back")
    p210d.reset()

    val p220d = Pad(channel = 2, width = 2)
    val around2 = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 2, 3, 0, 0,
      0, 0, 4, 5, 6, 0, 0,
      0, 0, 7, 8, 9, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 2, 3, 0, 0,
      0, 0, 4, 5, 6, 0, 0,
      0, 0, 7, 8, 9, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0
    )
    assert(p220d.forward(v2) == around2, "around2")
    assert(p220d.backward(p220d.forward(v2)) == v2, "around2 back")
    p220d.reset()


    // trans

    val t = DenseVector[Double](
      1, 2, 3,
      4, 5, 6,
      7, 8, 9
    )
    val t11u = Pad(channel = 1, width = 1, ud = "up")
    val trans1 = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 3, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 4, 0, 5, 0, 6, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 7, 0, 8, 0, 9, 0,
      0, 0, 0, 0, 0, 0, 0
    )
    assert(t11u.forward(t) == trans1, "trans1")
    assert(t11u.backward(t11u.forward(t)) == t, "trans1 back")
    t11u.reset()

    val t2 = DenseVector[Double](
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,

      1, 2, 3,
      4, 5, 6,
      7, 8, 9
    )
    val t21u = Pad(channel = 2, width = 1, ud = "up")
    val trans2 = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 3, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 4, 0, 5, 0, 6, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 7, 0, 8, 0, 9, 0,
      0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 3, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 4, 0, 5, 0, 6, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 7, 0, 8, 0, 9, 0,
      0, 0, 0, 0, 0, 0, 0
    )
    assert(t21u.forward(t2) == trans2, "trans2")
    assert(t21u.backward(t21u.forward(t2)) == t2, "trans2 back")
    t21u.reset()

    val t22u = Pad(channel = 2, width = 2, ud = "up")
    val trans22 = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    )
    assert(t22u.forward(t2) == trans22, "trans22")
    assert(t22u.backward(t22u.forward(t2)) == t2, "trans22 back")
    t21u.reset()


    println("all clear!")
  }
}


// }}}

