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

  // å¤ããfigãæ¸¡ããã¨ã§ãåä¸ã®ã¦ã£ã³ãã¦ã§ã°ã©ããæç»ããããã«ãªãã
  def Histgram2(fig: Figure, xs: Seq[DenseVector[Double]], row: Int = 1, filename: String = ""): Unit = {
    // ä»¥åã®æç»çµæããªã»ããããã
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

  def Plot(fig: Figure, xs: Seq[DenseVector[Double]], epoch: Int, row: Int = 1, filename: String = ""): Unit = {
    // ä»¥åã®æç»çµæããªã»ããããã
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
      plt.xlim = (0,epoch)
      // plt.ylim = (0,ymax+ymax*0.1)
      plt.title = s"${r*col+c}"
    }

    // fig.refresh

    if(filename != ""){
      fig.saveas(filename)
    }
  }
}
