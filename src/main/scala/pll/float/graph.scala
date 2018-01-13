package pll.float

import breeze.linalg._
import breeze.plot._
import typeAlias._
import CastImplicits._

object graph {
  def Histgram(xs: Seq[DenseVector[T]], row: Int = 1, filename: String = ""): Unit = {
    val col: Int = math.ceil(xs.size / row.toFloat).toInt
    val fig      = Figure()

    for {
      r <- 0 until row
      c <- 0 until col
      if (r * col + c < xs.size)
    } {
      val plt = fig.subplot(row, col, r * col + c)
      plt += hist(xs(r * col + c), 30)
    }

    if (filename != "") {
      fig.saveas(filename)
    }
  }

  // 外からfigを渡すことで、同一のウィンドウでグラフを描画するようになる。
  def Histgram2(fig: Figure, xs: Seq[DenseVector[T]], row: Int = 1, filename: String = ""): Unit = {
    // 以前の描画結果をリセットする。
    fig.clear

    val col: Int = math.ceil(xs.size / row.toFloat).toInt

    for {
      r <- 0 until row
      c <- 0 until col
      if (r * col + c < xs.size)
    } {
      val plt = fig.subplot(row, col, r * col + c)
      plt += hist(xs(r * col + c), 30)
      plt.title = s"hist ${r * col + c}"
    }

    if (filename != "") {
      fig.saveas(filename)
    }
  }

  def Line(fig: Figure, xs: Seq[DenseVector[T]], x_max: Int, row: Int = 1, filename: String = "")(
      implicit titles: Seq[String] = Seq()): Unit = {
    // 以前の描画結果をリセットする。
    fig.clear

    val x: DenseVector[T] = linspace(0, xs(0).size, xs(0).size)
    // val ymax = max(xs.map(max(_)))
    val col: Int = math.ceil(xs.size / row.toFloat).toInt

    for {
      r <- 0 until row
      c <- 0 until col
      if (r * col + c < xs.size)
    } {
      val plt = fig.subplot(row, col, r * col + c)
      plt += plot(x, xs(r * col + c))
      plt.xlim = (0, x_max)
      // plt.ylim = (0,ymax+ymax*0.1)
      if (titles.size != 0 && r * col + c < titles.size) {
        plt.title = s"${titles(r * col + c)}"
      }
      else {
        plt.title = s"${r * col + c}"
      }
    }

    // fig.refresh

    if (filename != "") {
      fig.saveas(filename)
    }
  }

  def Image(fig: Figure, xs: Seq[DenseMatrix[T]], row: Int = 1, filename: String = ""): Unit = {
    // 以前の描画結果をリセットする。
    fig.clear

    def upsideDown(mat: DenseMatrix[T]): DenseMatrix[T] = {
      val buf = DenseMatrix.zeros[T](mat.rows, mat.cols)
      (0 until mat.rows).foreach(r => buf(r, ::) := mat(mat.rows - 1 - r, ::))

      buf
    }

    // val ymax = max(xs.map(max(_)))
    val col: Int = math.ceil(xs.size / row.toFloat).toInt

    for {
      r <- 0 until row
      c <- 0 until col
      if (r * col + c < xs.size)
    } {
      val plt = fig.subplot(row, col, r * col + c)
      plt += image(convert(upsideDown(xs(r * col + c)), Double)) /* breeze.plot.image method's first argument is DenseMatrix[Double]. */

      plt.xlim = (0, xs(r * col + c).rows)
      plt.ylim = (0, xs(r * col + c).cols)
      // plt.title = s"${r*col+c}"
    }

    // fig.refresh

    if (filename != "") {
      fig.saveas(filename)
    }
  }
}
