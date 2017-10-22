package pll


import breeze.linalg._
import breeze.plot._

object graph {
  def Histgram(xs: Seq[DenseVector[Double]], row: Int = 1, filename: String = ""): Unit = {
    val col: Int = math.ceil(xs.size / row.toDouble).toInt
    val fig = Figure()

    for {
      r <- 0 until row
      c <- 0 until col
      if (r*col+c < xs.size)
    } {
      val plt = fig.subplot(row, col, r*col+c)
      plt += hist(xs(r*col+c), 30)
    }

    if(filename != ""){
      fig.saveas(filename)
    }
  }

  // 外からfigを渡すことで、同一のウィンドウでグラフを描画するようになる。
  def Histgram2(fig: Figure, xs: Seq[DenseVector[Double]], row: Int = 1, filename: String = ""): Unit = {
    // 以前の描画結果をリセットする。
    fig.clear

    val col: Int = math.ceil(xs.size / row.toDouble).toInt

    for {
      r <- 0 until row
      c <- 0 until col
      if (r*col+c < xs.size)
    } {
      val plt = fig.subplot(row, col, r*col+c)
      plt += hist(xs(r*col+c), 30)
      plt.title = s"hist ${r*col+c}"
    }

    if(filename != ""){
      fig.saveas(filename)
    }
  }

  def Plot(fig: Figure, xs: Seq[DenseVector[Double]], x_max: Int, row: Int = 1, filename: String = ""): Unit = {
    // 以前の描画結果をリセットする。
    fig.clear

    val x = linspace(0,xs(0).size,xs(0).size)
    // val ymax = max(xs.map(max(_)))
    val col: Int = math.ceil(xs.size / row.toDouble).toInt

    for {
      r <- 0 until row
      c <- 0 until col
      if (r*col+c < xs.size)
    } {
      val plt = fig.subplot(row, col, r*col+c)
      plt += plot(x,xs(r*col+c))
      plt.xlim = (0,x_max)
      // plt.ylim = (0,ymax+ymax*0.1)
      plt.title = s"${r*col+c}"
    }

    // fig.refresh

    if(filename != ""){
      fig.saveas(filename)
    }
  }

  def Image(fig: Figure, xs: Seq[DenseMatrix[Double]], row: Int = 1, filename: String = ""): Unit = {
    // 以前の描画結果をリセットする。
    fig.clear

    def upsideDown(mat: DenseMatrix[Double]) = {
      val buf = DenseMatrix.zeros[Double](mat.rows, mat.cols)
      for {
        r <- 0 until mat.rows
        c <- 0 until mat.cols
      } {
        buf(r, c) = mat(mat.rows-1-r, c)
      }
      buf
    }

    // val ymax = max(xs.map(max(_)))
    val col: Int = math.ceil(xs.size / row.toDouble).toInt

    for {
      r <- 0 until row
      c <- 0 until col
      if (r*col+c < xs.size)
    } {
      val plt = fig.subplot(row, col, r*col+c)
      plt += image(upsideDown(xs(r*col+c)))
      plt.xlim = (0,xs(r*col+c).rows)
      plt.ylim = (0,xs(r*col+c).cols)
      // plt.title = s"${r*col+c}"
    }

    // fig.refresh

    if(filename != ""){
      fig.saveas(filename)
    }
  }
}
