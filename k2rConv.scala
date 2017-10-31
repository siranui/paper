// package pll


import pll._
import breeze.linalg._

/** convolution class.
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
  input_width: Int,
  filter_width: Int,
  filter_set: Int = 1,
  channel: Int = 1,
  stride: Int = 1,
  distr: String = "Gaussian",
  SD: Double = 0.1,
  update_method: String = "SGD",
  lr: Double = 0.01
) extends Layer {

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

    val xmat = xs.map(_.toDenseMatrix).reduceLeft(DenseMatrix.vertcat(_,_))
    val Fmat =
      F.map(_.map(_.toDenseMatrix).reduce(DenseMatrix.vertcat(_,_)))
        .reduce(DenseMatrix.vertcat(_,_))
        .reshape(channel, filter_width*filter_width*filter_set/* = k^2*M */).t


    val res = Fmat * xmat

    val buf = DenseMatrix.zeros[Double](filter_set, input_width*input_width)
    ((0 until filter_width*filter_width) zip (for(i <- 0 until input_width*input_width if(i % input_width < filter_width)) yield i)).foreach{ case (idx,num) =>
      buf +=  rotate.left(
        num,
        res(idx*filter_set until idx*filter_set+filter_set,::)
      )
    }

    val fx = buf(*,::).map{ i =>
      val j = reshape(i.t, input_width, input_width)
      j(0 until out_width, 0 until out_width).t.toDenseVector
    }.t.toDenseVector
    fx + B.reduce(DenseVector.vertcat(_, _))
  }

  def backward(d: DVD): DVD = {
    val ds: ADVD = utils.divideIntoN(d, N = filter_set)
    val dmat = ds.map{ m =>
      padRight(reshape(m, out_width, out_width), (input_width,   input_width), 0d).reshape(1,input_width * input_width)
    }.reduceLeft(DenseMatrix.vertcat(_,_))

    // update: bias's update value
    assert(dB.length == ds.length)
    for (i <- dB.indices) {
      assert(dB(i).size == ds(i).size)
      dB(i) += ds(i)
    }

    assert(this.xs.nonEmpty)
    val xs = this.xs.get.head
    this.xs = Some(this.xs.get.tail)
    val xmat = xs.map(_.toDenseMatrix).reduceLeft(DenseMatrix.vertcat(_,_))
    val Fmat =
      F.map(_.map(_.toDenseMatrix).reduce(DenseMatrix.vertcat(_,_)))
        .reduce(DenseMatrix.vertcat(_,_))
        .reshape(channel, filter_width*filter_width*filter_set).t

    // println(s"Fmat:\n$Fmat\n")
    // println(s"dmat:\n$dmat\n")
    // (Fmat * dmat).t.toDenseVector
    val res = Fmat * xmat

    val buf = DenseMatrix.zeros[Double](filter_set, input_width*input_width)
    ((0 until filter_width*filter_width) zip (for(i <- 0 until input_width*input_width if(i % input_width < filter_width)) yield i)).foreach{ case (idx,num) =>
      buf +=  rotate.left(
        num,
        res(idx*filter_set until idx*filter_set+filter_set,::)
      )
    }

    val fx = buf(*,::).map{ i =>
      val j = reshape(i.t, input_width, input_width)
      j(0 until out_width, 0 until out_width).t.toDenseVector
    }.t.toDenseVector
    fx
    buf.t.toDenseVector
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
    val fos = new java.io.FileOutputStream(fn, true)
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
      F(fs)(ch)(v) = str(0)(fs * F(fs).length + ch * F(fs)(ch).size + v)
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

  // Allow only "SQUARE"" images and filters.
  def filter2Weight(filter: DVD, input_size: Int, stride: Int): DMD = {
    val w: Int = math.sqrt(input_size).toInt
    val h: Int = math.sqrt(filter.length).toInt
    val out_w: Int = (math.floor((w - h) / stride) + 1).toInt

    // create Weight matrix
    val W = DenseMatrix.zeros[Double](math.pow(out_w, 2).toInt, math.pow(w, 2).toInt)
    for {
      i <- 0 until out_w
      j <- 0 until out_w
      p <- 0 until h
      q <- 0 until h
    } {
      val W_row = i * out_w + j
      val W_col = (i * stride + p) * w + stride * j + q
      W(W_row, W_col) = filter(p * h + q)
    }

    W
  }

  /*
   * filters[ch, height*width]
   */
  def filter2Weight(filters: ADVD, input_size: Int, stride: Int = 1): DMD = {
    val Ws = filters.map(i => filter2Weight(i, input_size, stride))
    Ws.reduceLeft(DenseMatrix.horzcat(_, _))
  }

  def Weight2filter(dmat: DMD, filter_size: Int, stride: Int = 1): DVD = {
    val h = math.sqrt(filter_size).toInt
    val out_w = math.sqrt(dmat.rows).toInt
    val w = (out_w - 1) * stride + h

    val Filter = DenseVector.zeros[Double](filter_size)
    for {
      i <- 0 until out_w
      j <- 0 until out_w
      p <- 0 until h
      q <- 0 until h
    } {
      val d_row = i * out_w + j
      val d_col = (i * stride + p) * w + stride * j + q
      Filter(p * h + q) += dmat(d_row, d_col)
    }

    Filter
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

// test
object k2rConvTest {

  def o(n:Int) = DenseVector.ones[Double](n)
  def r(n:Int) = convert(DenseVector.range(0, n), Double)
  val (f, x, d) = (r(4), r(9)+ 1d, r(4))

  def k2r() = {
    val k2r = k2rConv(3,2)

    k2r.F = Array(Array(f))

    println("forward:")
    println(k2r.forward(x))
    println("backward:")
    println(k2r.backward(d))
  }

  def normal() = {
    val conv = pll.Convolution(3,2)

    conv.F = Array(Array(f))

    println("forward:")
    println(conv.forward(x))
    println("backward:")
    println(conv.backward(d))

  }

  def main(args: Array[String]) {

    if(args.size != 0){
      args(0) match {
        case "k2r"        => k2r()
        case "normal" | _ => normal()
      }
    } else {
      normal()
    }

  }
}

