package pll

//import pll._
import breeze.linalg._

/**
 * convolution class.
 *
 * @note Don't support "rectangle"(width != height).
 *       Only "perfect square"(width == height) is supported.
 *
 * @param input_width   width(= height) of input image.
 * @param filter_width  width(= height) of filter.
 * @param filter_set    num of filters.
 * @param channel       channel of input image.
 *                      e.g.)  RGB image -> 3 channel
 * @param stride        stride size. Don't set "zero" or "minus".
 * @param distr         distribution
 * @param SD            initial weight's standard deviation.
 * @param update_method update method for learning parameters.
 * @param lr            learning rate.
 */
case class k2rConv(
    input_width:   Int,
    filter_width:  Int,
    filter_set:    Int    = 1,
    channel:       Int    = 1,
    stride:        Int    = 1,
    distr:         String = "Gaussian",
    SD:            Double = 0.1,
    update_method: String = "SGD",
    lr:            Double = 0.01) extends Layer {

  type DVD = DenseVector[Double]
  type DMD = DenseMatrix[Double]
  type ADVD = Array[DVD]
  type ADMD = Array[DMD]

  assert(stride >= 1, "stride must 1 or over.")

  val opt_filter: Opt = Opt.create(update_method, lr)
  val opt_bias: Opt = Opt.create(update_method, lr)

  var xs: Option[List[ADVD]] = None //storage: input by channel
  var Ws: Option[Array[ADMD]] = None

  val out_width: Int = (math.floor((input_width - filter_width) / stride) + 1).toInt

  // filter
  var F: Array[ADVD] = filter_init(filter_set, channel, filter_width * filter_width)
  // bias
  var B: ADVD = Array.ofDim[DVD](filter_set)
    .map(_ => DenseVector.zeros[Double](out_width * out_width))

  var dF: Array[ADVD] = F.map(_.map(_ => DenseVector.zeros[Double](filter_width * filter_width)))
  var dB: ADVD = B.map(_ => DenseVector.zeros[Double](out_width * out_width))

  //F.map(opt.register(_))
  opt_filter.register(F.flatten)
  opt_bias.register(B)

  def forward(x: DVD): DVD = {
    val xs: ADVD = utils.divideIntoN(x, N = channel)
    this.xs = Some(xs :: this.xs.getOrElse(Nil)) //store: input

    val xmat = xs.map(_.toDenseMatrix).reduceLeft(DenseMatrix.vertcat(_, _))
    val Fmat =
      F.map(_.map(_.toDenseMatrix).reduce(DenseMatrix.vertcat(_, _)))
        .reduce(DenseMatrix.vertcat(_, _))
        .reshape(channel, filter_width * filter_width * filter_set /* = k^2*M */ ).t

    val res = Fmat * xmat

    val buf = DenseMatrix.zeros[Double](filter_set, input_width * input_width)
    ((0 until filter_width * filter_width) zip (for (i <- 0 until input_width * input_width if (i % input_width < filter_width)) yield i)).foreach{
      case (idx, num) =>
        buf += rotate.left(
          num,
          res(idx * filter_set until idx * filter_set + filter_set, ::))
    }

    val fx = buf(*, ::).map{ i =>
      val j = reshape(i.t, input_width, input_width)
      j(0 until out_width, 0 until out_width).t.toDenseVector
    }.t.toDenseVector
    fx + B.reduce(DenseVector.vertcat(_, _))
  }

  def backward(d: DVD): DVD = {

    val ds: ADVD = utils.divideIntoN(d, N = filter_set)
    val dmat = ds.map{ m =>
      padRight(reshape(m, out_width, out_width), (input_width, input_width), 0d).reshape(1, input_width * input_width)
    }.reduceLeft(DenseMatrix.vertcat(_, _))

    // update: bias's update value
    assert(dB.length == ds.length)
    for (i <- dB.indices) {
      assert(dB(i).size == ds(i).size)
      dB(i) += ds(i)
    }

    // update: Filter's update value
    assert(this.xs.nonEmpty)
    val xs = this.xs.get.head
    this.xs = Some(this.xs.get.tail)
    val xmat = xs.map(_.toDenseMatrix).reduceLeft(DenseMatrix.vertcat(_, _))
    val xT = xmat.t

    val dw_tmp = (for ((idx, num) <- ((0 until filter_width * filter_width) zip (for (i <- 0 until input_width * input_width if (i % input_width < filter_width)) yield i))) yield {
      val tmp = dmat * rotate.up(num, xT)
      tmp.t.reshape(filter_set * channel, 1)
    }).reduce(DenseMatrix.horzcat(_, _))

    for { i <- 0 until filter_set; j <- 0 until channel } {
      dF(i)(j) += dw_tmp(i * channel + j, ::).t
    }

    // calculate: return value( Weight^T * dOut )
    val Fmat =
      F.map(_.map(_.toDenseMatrix).reduce(DenseMatrix.vertcat(_, _)))
        .reduce(DenseMatrix.vertcat(_, _))
        .reshape(channel, filter_width * filter_width * filter_set).t

    val FmatT = (for (i <- 0 until filter_width * filter_width) yield {
      Fmat(i * filter_set until i * filter_set + filter_set, ::).t
    }).reduce(DenseMatrix.vertcat(_, _))

    val dx_pre_shift = FmatT * dmat

    val dx = DenseMatrix.zeros[Double](channel, input_width * input_width)
    ((0 until filter_width * filter_width) zip (for (i <- 0 until input_width * input_width if (i % input_width < filter_width)) yield i)).foreach{
      case (idx, num) =>
        dx += rotate.right(
          num,
          dx_pre_shift(idx * channel until idx * channel + channel, ::))
    }

    dx.t.toDenseVector
  }

  def update(): Unit = {
    val wf = opt_filter.update(F.flatten, dF.flatten)
    for {
      i <- F.indices
      j <- F(i).indices
    } {
      F(i)(j) -= wf(i * F(i).length + j)
    }

    val wb = opt_bias.update(B, dB)
    B -= wb

    reset()
  }

  def reset(): Unit = {
    xs = None
    Ws = None
    dF = dF.map(_.map(_ => DenseVector.zeros[Double](filter_width * filter_width)))
    dB = dB.map(_ => DenseVector.zeros[Double](out_width * out_width))
  }

  def save(fn: String): Unit = {
    val fos = new java.io.FileOutputStream(fn, false)
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw = new java.io.PrintWriter(osw)

    val flatF = F.flatMap(_.map(_.toArray)).flatten
    pw.write(flatF.mkString(","))
    pw.write("\n")

    val flatB = B.flatMap(_.toArray)
    pw.write(flatB.mkString(","))
    pw.write("\n")

    pw.close()
  }

  def load(fn: String) {
    val str = io.Source.fromFile(fn).getLines.map(_.split(",").map(_.toDouble)).toArray

    // set 'F' parameter
    for {
      fs <- F.indices
      ch <- F(fs).indices
      v <- 0 until F(fs)(ch).size
    } {
      F(fs)(ch)(v) = str(0)(fs * F(fs).length * F(fs)(ch).size + ch * F(fs)(ch).size + v)
    }

    // set 'B' parameter
    for {
      fs <- B.indices
      v <- F(fs).indices
    } {
      B(fs)(v) = str(1)(fs * B(fs).size + v)
    }
  }

  override def load(data: List[String]): List[String] = {
    val flst = data(0).split(",").map(_.toDouble)
    val blst = data(1).split(",").map(_.toDouble)

    // set 'F' parameter
    for {
      fs <- F.indices
      ch <- F(fs).indices
      v <- 0 until F(fs)(ch).size
    } {
      F(fs)(ch)(v) = flst(fs * F(fs).length + ch * F(fs)(ch).size + v)
    }

    // set 'B' parameter
    for {
      fs <- B.indices
      v <- F(fs).indices
    } {
      B(fs)(v) = blst(fs * B(fs).size + v)
    }

    data.drop(2)
  }

  // {{{ helper

  def filter_init(M: Int, K: Int, H: Int): Array[ADVD] = {
    val filters: Array[ADVD] = Array.ofDim[DVD](M, K)
    for {
      i <- filters.indices
      j <- filters(i).indices
    } {
      filters(i)(j) = distr match {
        case "Xavier"       => Xavier(H, input_width * input_width)
        case "He"           => He(H, input_width * input_width)
        case "Uniform"      => Uniform(H, SD)
        case "Gaussian" | _ => Gaussian(H, SD)
      }
    }

    filters
  }

  def copy_F(): Array[ADVD] = F.map(_.map(_.copy))

  def copy_B(): ADVD = B.map(_.copy)

  override def duplicate(): Convolution = {
    val dup = Convolution(input_width, filter_width, filter_set, channel, stride, distr, SD, update_method, lr)
    dup.F = this.copy_F()
    dup.B = this.copy_B()
    dup
  }

  // }}}

}

object rotate {
  import breeze.linalg._
  def left[T](n: Int, s: Seq[T]) = s.drop(n % s.size) ++ s.take(n % s.size)
  def right[T](n: Int, s: Seq[T]) = s.takeRight(n % s.size) ++ s.dropRight(n % s.size)

  def left(n: Int, v: DenseVector[Double]) = DenseVector.vertcat(v(n % v.size until v.size), v(0 until n % v.size))
  def right(n: Int, v: DenseVector[Double]) = DenseVector.vertcat(v(v.size - n % v.size until v.size), v(0 until v.size - n % v.size))

  def left(n: Int, m: DenseMatrix[Double]) = DenseMatrix.horzcat(m(::, n % m.cols until m.cols), m(::, 0 until n % m.cols))
  def right(n: Int, m: DenseMatrix[Double]) = DenseMatrix.horzcat(m(::, m.cols - n % m.cols until m.cols), m(::, 0 until m.cols - n % m.cols))
  def up(n: Int, m: DenseMatrix[Double]) = DenseMatrix.vertcat(m(n % m.rows until m.rows, ::), m(0 until n % m.rows, ::))
  def down(n: Int, m: DenseMatrix[Double]) = DenseMatrix.vertcat(m(m.rows - n % m.rows until m.rows, ::), m(0 until m.rows - n % m.rows, ::))
}

// test
object k2rConvTest {

  def o(n: Int) = DenseVector.ones[Double](n)
  def r(n: Int) = convert(DenseVector.range(0, n), Double)
  val in_w = 32
  val fil_w = 7
  val stride = 1
  val out_w_s = utils.out_width(in_w, fil_w, stride)
  val out_w = in_w - fil_w + 1
  val f_set = 8
  val ch = 6
  val (f, x, d, ds) = (r(fil_w * fil_w), r(in_w * in_w * ch) + 1d, r(out_w * out_w * f_set), r(out_w_s * out_w_s * f_set))

  def normal() = {
    val conv = pll.Convolution(in_w, fil_w, f_set, ch, stride)

    // conv.F = Array(Array(f,f),Array(f,f),Array(f,f))
    conv.F = conv.F.map(_.map(_ => f))

    // println("forward:")
    val fw = conv.forward(x)
    // println(fw)
    // println("backward:")
    val bw = conv.backward(ds)
    // println(bw)
    // println("dF:")
    // println(conv.dF.flatten.reduce(DenseVector.vertcat(_,_)))

  }

  def k2r() = {
    val k2r = k2rConv(in_w, fil_w, f_set, ch)

    k2r.F = k2r.F.map(_.map(_ => f))

    // println("forward:")
    val fw = k2r.forward(x)
    // println(fw)
    // println("backward:")
    val bw = k2r.backward(d)
    // println(bw)
    // println("dF:")
    // println(k2r.dF.flatten.reduce(DenseVector.vertcat(_,_)))
  }

  def i2c() = {
    val i2c = i2cConv(in_w, fil_w, f_set, ch, stride)

    i2c.F = i2c.F.map(_.map(_ => f))

    // println("forward:")
    val fw = i2c.forward(x)
    // println(fw)
    // println("backward:")
    val bw = i2c.backward(ds)
    // println(bw)
    // println("dF:")
    // println(k2r.dF.flatten.reduce(DenseVector.vertcat(_,_)))
  }

  def main(args: Array[String]) {

    if (args.size != 0) {
      args(0) match {
        case "normal" => utils.printExcutingTime(normal())
        case "k2r"    => utils.printExcutingTime(k2r())
        case "i2c"    => utils.printExcutingTime(i2c())
        case _ =>
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
          println("normal:")
          utils.printExcutingTime(normal())
          println("kn2row:")
          utils.printExcutingTime(k2r())
          println("im2col:")
          utils.printExcutingTime(i2c())
          println("--")
      }
    }
    else {
      normal()
    }

  }
}
