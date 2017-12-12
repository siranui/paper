package font_GLO

import pll._
import breeze.linalg._

class font_GLO extends Network { // {{{

  // 100 -> 25*2*2 -> 10*4*4 -> 5*8*8 -> 3*16*16 -> 1*32*32
  //
  // case class Pad(channel: Int, width: Int, ud: String = "down") extends Layer{
  //
  // case class Convolution(
  //   input_width: Int,
  //   filter_width: Int,
  //   filter_set: Int = 1,
  //   channel: Int = 1,
  //   stride: Int = 1,
  //   distr: String = "Gaussian",
  //   SD: Double = 1d,
  //   update_method: String = "SGD",
  //   lr: Double = 0.01,
  // ) extends Layer {
  // out_width = (math.floor((input_width - filter_width) / stride) + 1).toInt
  //

  // val pad = 3
  // val fil_w = 4
  // val stride = 2
  val pad    = 1
  val fil_w  = 2
  val stride = 1
  val lr     = 0d

  layers =
    (new Pad(25, pad, "up")) ::
      (new Convolution(2 * (pad + 1) + pad, fil_w, 10, 25, stride, "He", 1d, "Adam", lr)) ::
      (new LeakyReLU(0.01)) ::
      (new Pad(10, pad, "up")) ::
      (new Convolution(4 * (pad + 1) + pad, fil_w, 5, 10, stride, "He", 1d, "Adam", lr)) ::
      (new LeakyReLU(0.01)) ::
      (new Pad(5, pad, "up")) ::
      (new Convolution(8 * (pad + 1) + pad, fil_w, 3, 5, stride, "He", 1d, "Adam", lr)) ::
      (new LeakyReLU(0.01)) ::
      (new Pad(3, pad, "up")) ::
      (new Convolution(16 * (pad + 1) + pad, fil_w, 1, 3, stride, "He", 1d, "Adam", lr)) ::
      (new LeakyReLU(0.01)) ::
      layers

  val Lap = Convolution(32, 3, 1, 1, 1, "", 1d, "", 1d)
  val l = DenseVector[Double](
    1, 1, 1, 1, -8, 1, 1, 1, 1
  )
  Lap.F = Array(Array(l))
  def calc_Lap_loss(y: DenseVector[Double], t: DenseVector[Double]) = {
    math.pow(2d, -2d) * sum((Lap.forward(y) - Lap.forward(t)).map(math.abs))
  }
  def calc_Lap_grad(y: DenseVector[Double], t: DenseVector[Double]) = {
    val o = Lap.forward(y) - Lap.forward(t)
    //val tmp = DenseVector.zeros[Double](y.size)
    val W = Lap.filter2Weight(l, 32 * 32, 1)
    W.t * o.map(a => if (a > 0) 1d else -1d)
  }

} // }}}

object font_GLO {
  def main(args: Array[String]) {
    val rand  = new util.Random(0)
    val ds    = 50
    val epoch = 100
    val batch = 1

    var Z = Array.ofDim[DenseVector[Double]](ds)
    Z = Z.map(dv => DenseVector.fill(100) { rand.nextGaussian })
    val HOME    = sys.env("HOME")
    val train_d = read(s"$HOME/fonts/tmp/AGENCYB/AGENCYB-d.txt", ds)
    // val train_d = read_labo("data/cifar10/train-d.txt", ds)

    val dataset: Array[(DenseVector[Double], DenseVector[Double])] = Z zip train_d

    val g = new font_GLO()

    // training
    println("--- training start ---")
    for (e <- 0 until epoch) {
      var E   = 0d
      var cnt = 0

      for ((z, x) <- dataset) {
        val y = g.predict(z)
        E += g.calc_Lap_loss(y, x)
        val d = g.calc_Lap_grad(y, x)
        // E += g.calc_L2(y,x)
        // val d = g.calc_L2_grad(y,x)
        g.update(d)

        cnt += 1
      }

      // output
      if (e == 0 || e % 100 == 0 || e == epoch - 1) {
        var ys = List[DenseVector[Int]]()
        for ((z, x) <- dataset) {
          val y = g.predict(z).map(o => (o * 256).toInt)
          ys = y :: ys
          g.reset()
        }
        write(
          s"src/main/scala/GLO/results/font_GLO_ds${ds}_epoch${e}of${epoch}_batch${batch}_pad${g.pad}_stride${g.stride}.txt",
          ys.reverse)
      }

      println(s"$e, $E")
    }

    // check output dimention
    // println(g.predict(dataset(0)._1).size)

  }

  def read(fn: String, ds: Int = 100): Array[DenseVector[Double]] = { // {{{
    val f = io.Source
      .fromFile(fn)
      .getLines
      .take(ds)
      .map(_.split(",").map(_.toDouble / 256d).toArray)
      .toArray
    val g = f.map(a => DenseVector(a))
    g
  } // }}}

  def read_labo(fn: String, ds: Int = 100): Array[DenseVector[Double]] = { // {{{
    val f = io.Source
      .fromFile(fn)
      .getLines
      .take(ds)
      .map(_.split(",").map(_.toDouble / 256d).toArray)
      .toArray
    val r = f.map(i => DenseVector((for (idx <- 0 until i.size by 3) yield i(idx)).toArray))
    val g = f.map(i => DenseVector((for (idx <- 1 until i.size by 3) yield i(idx)).toArray))
    val b = f.map(i => DenseVector((for (idx <- 2 until i.size by 3) yield i(idx)).toArray))
    val rgb = (for (idx <- 0 until r.size) yield {
      DenseVector.vertcat(DenseVector.vertcat(r(idx), g(idx), b(idx)))
    }).toArray
    rgb
  } // }}}

  def write(fn: String, dataList: List[DenseVector[Int]]) { // {{{
    val fos = new java.io.FileOutputStream(fn, false) //true: 追記, false: 上書き
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw  = new java.io.PrintWriter(osw)
    for (data <- dataList) {
      for (i <- 0 until data.size) {
        pw.write(data(i).toString)
        if (i != data.size - 1) {
          pw.write(",")
        }
      }
      pw.write("\n")
    }
    pw.close()
  } // }}}

}
