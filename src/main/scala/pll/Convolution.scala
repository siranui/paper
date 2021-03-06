package pll

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
case class Convolution(
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

  type DVD  = DenseVector[Double]
  type DMD  = DenseMatrix[Double]
  type ADVD = Array[DVD]
  type ADMD = Array[DMD]

  assert(stride >= 1, "stride must 1 or over.")

  val opt_filter: Opt = Opt.create(update_method, lr)
  val opt_bias: Opt   = Opt.create(update_method, lr)

  var xs: Option[List[ADVD]]  = None //storage: input by channel
  var Ws: Option[Array[ADMD]] = None

  val out_width: Int = (math.floor((input_width - filter_width) / stride) + 1).toInt

  // filter
  var F: Array[ADVD] = filter_init(filter_set, channel, filter_width * filter_width)
  // bias
  var B: ADVD = Array
    .ofDim[DVD](filter_set)
    .map(_ => DenseVector.zeros[Double](out_width * out_width))

  var dF: Array[ADVD] = F.map(_.map(_ => DenseVector.zeros[Double](filter_width * filter_width)))
  var dB: ADVD        = B.map(_ => DenseVector.zeros[Double](out_width * out_width))

  //F.map(opt.register(_))
  opt_filter.register(F.flatten)
  opt_bias.register(B)

  def forward(x: DVD): DVD = {
    val xs: ADVD = utils.divideIntoN(x, N = channel)
    this.xs = Some(xs :: this.xs.getOrElse(Nil)) //store: input

    val Ws = Array.ofDim[DMD](filter_set, channel)

    val u = for (fs <- 0 until filter_set) yield {
      val m = for (ch <- 0 until channel) yield {
        Ws(fs)(ch) = filter2Weight(F(fs)(ch), input_width * input_width, stride)
        Ws(fs)(ch) * xs(ch) // = W * x
      }
      m.fold(B(fs))(_ + _) // 1 feature map
    }

    this.Ws = Some(Ws)

    u.reduceLeft((i, j) => DenseVector.vertcat(i, j)) // convert to 1d
  }

  def backward(d: DVD): DVD = {
    val dmap: ADVD = utils.divideIntoN(d, N = filter_set)

    assert(dB.length == dmap.length)
    for (i <- dB.indices) {
      assert(dB(i).size == dmap(i).size)
      dB(i) += dmap(i)
    }

    assert(this.xs.nonEmpty)
    val xs = this.xs.get.head
    this.xs = Some(this.xs.get.tail)
    // dWs(filter_set, channel)
    val dWs: Array[ADMD] = dmap.map(d => xs.map(x => d * x.t))

    for { i <- dWs.indices; j <- dWs(i).indices } {
      dF(i)(j) += Weight2filter(dWs(i)(j), filter_width * filter_width, stride)
    }

    val Ws = this.Ws.get
    val dx: DVD = (for { fs <- 0 until filter_set } yield {
      (for { ch <- 0 until channel } yield {
        val tmp: DVD = (dmap(fs).t * Ws(fs)(ch)).t
        reshape(tmp.t, input_width, input_width).t.toDenseVector
      }).reduce(DenseVector.vertcat(_, _))
    }).reduce(_ + _)

    dx
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
    val pw  = new java.io.PrintWriter(osw)

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
      v  <- 0 until F(fs)(ch).size
    } {
      F(fs)(ch)(v) = str(0)(fs * F(fs).length * F(fs)(ch).size + ch * F(fs)(ch).size + v)
    }

    // set 'B' parameter
    for {
      fs <- B.indices
      v  <- F(fs).indices
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
      v  <- 0 until F(fs)(ch).size
    } {
      F(fs)(ch)(v) = flst(fs * F(fs).length + ch * F(fs)(ch).size + v)
    }

    // set 'B' parameter
    for {
      fs <- B.indices
      v  <- F(fs).indices
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
    val w: Int     = math.sqrt(input_size).toInt
    val h: Int     = math.sqrt(filter.length).toInt
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
    val h     = math.sqrt(filter_size).toInt
    val out_w = math.sqrt(dmat.rows).toInt
    val w     = (out_w - 1) * stride + h

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
    val dup = Convolution(
      input_width,
      filter_width,
      filter_set,
      channel,
      stride,
      distr,
      SD,
      update_method,
      lr)
    dup.F = this.copy_F()
    dup.B = this.copy_B()
    dup
  }

  // }}}

}

// {{{ test
object ConvolutionTest {
  def main(args: Array[String]) {

    // Convolution(
    //    f: ActivateFunction,
    //    filter_width: Int, filter_set: Int = 1,
    //    channel: Int = 1, stride: Int = 1 )
    val fH                    = 3
    val fS                    = 2
    val ch                    = 3
    val stride                = 2
    val input_width           = 10
    val distr: String         = "Xavier"
    val SD: Double            = 0d
    val update_method: String = "AdaGrad"
    val lr: Double            = 0.01
    val conv                  = Convolution(input_width, fH, fS, ch, stride, distr, SD, update_method, lr)
    // val img = DenseVector.rand(ch * input_width * input_width)
    val img   = DenseVector.range(0, ch * input_width * input_width).map(_.toDouble)
    val conv1 = conv.forward(img)
    println(conv1)
    println(s"length = ${conv1.size}")
    val back = conv.backward(conv1)
    println(s"img.size = ${img.size}")
    println(s"back.size = ${back.size}")
  }
}
// }}}
