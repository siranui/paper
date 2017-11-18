package pll

import breeze.linalg._

object Im2Col {
  type T = Double
  type DV = DenseVector[T]
  type DM = DenseMatrix[T]

  def im2col(x: Array[DV], fil_h: Int, fil_w: Int, ch: Int = 1, stride: Int = 1) = {
    val im: Array[Array[DV]] = x.map(b => utils.divideIntoN(b, ch))
    val in_w = math.sqrt(im(0)(0).size).toInt
    val images: Array[Array[DM]] = im.map(_.map(i => reshape(i, in_w, in_w).t))

    val out_w = utils.out_width(in_w, fil_w, stride)

    val col =
      (for (image <- images) yield {
        (for (image_ch <- image) yield {
          (for (i <- 0 until out_w; j <- 0 until out_w) yield {
            val m = image_ch(i * stride until i * stride + fil_h, j * stride until j * stride + fil_w).t
            m.reshape(fil_h * fil_w, 1)
          }).reduce(DenseMatrix.horzcat(_, _))
        }).reduce(DenseMatrix.vertcat(_, _))
      }).reduce(DenseMatrix.horzcat(_, _))

    col
  }

  def fil2mat(F: Array[Array[DV]]): DM = {
    val FF = F.map(f => f.reduce(DenseVector.vertcat(_, _)))
    FF.map(_.toDenseMatrix).reduce(DenseMatrix.vertcat(_, _))
  }

  case class Shape(N: Int, C: Int, H: Int, W: Int)

  def col2im(col: DM, x_shape: Shape, fil_h: Int, fil_w: Int, stride: Int = 1): Array[Array[DM]] = {
    val batch = x_shape.N
    val ch = x_shape.C
    val in_h = x_shape.H
    val in_w = x_shape.W
    val out_h = utils.out_width(in_h, fil_h, stride)
    val out_w = utils.out_width(in_w, fil_w, stride)
    val fil_size = fil_h * fil_w

    val im = Array.fill(batch, ch)(DenseMatrix.zeros[Double](in_h, in_w))
    for {
      b <- 0 until batch
      c <- 0 until ch
      oh <- 0 until out_h
      ow <- 0 until out_w
    } {
      im(b)(c)(
        oh * stride until oh * stride + fil_h,
        ow * stride until ow * stride + fil_w) += reshape(
          col(c * fil_size until c * fil_size + fil_size, b * out_h * out_w + oh * out_w + ow),
          fil_w,
          fil_h).t
    }
    im
  }

}

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
case class i2cConv(
    input_width:   Int,
    filter_width:  Int,
    filter_set:    Int    = 1,
    channel:       Int    = 1,
    stride:        Int    = 1,
    distr:         String = "Gaussian",
    SD:            Double = 0.1,
    update_method: String = "SGD",
    lr:            Double = 0.01) extends Layer {

  import Im2Col._

  type DVD = DenseVector[Double]
  type DMD = DenseMatrix[Double]
  type ADVD = Array[DVD]
  type ADMD = Array[DMD]

  assert(stride >= 1, "stride must 1 or over.")

  val opt_filter: Opt = Opt.create(update_method, lr)
  val opt_bias: Opt = Opt.create(update_method, lr)

  val out_width: Int = utils.out_width(input_width, filter_width, stride)

  var col_X: Option[DMD] = None //input that change for matrix
  var col_W: Option[DMD] = None //filter that change for matrix

  // filter
  var F: Array[ADVD] = filter_init(filter_set, channel, filter_width * filter_width)
  // bias
  var B: DVD = DenseVector.zeros[Double](filter_set)

  var dW: DMD = DenseMatrix.zeros[Double](filter_set, filter_width * filter_width * channel)
  var dB: DVD = DenseVector.zeros[Double](filter_set)

  opt_filter.register(Array(fil2mat(F)))
  opt_bias.register(Array(B))

  def forward(x: DVD): DVD = {
    val col = im2col(Array(x), filter_width, filter_width, channel, stride)
    val col_W = fil2mat(F)

    this.col_X = Some(col)
    this.col_W = Some(col_W)

    val outMat = col_W * col
    val Wx_plus_B = outMat(::, *) + B
    Wx_plus_B.t.toDenseVector
  }

  override def forwards(x: ADVD): ADVD = {
    val col = im2col(x, filter_width, filter_width, channel, stride)
    val col_W = fil2mat(F)

    this.col_X = Some(col)
    this.col_W = Some(col_W)

    val outMat = col_W * col
    val Wx_plus_B = outMat(::, *) + B

    // println(s"debug: outMat(${outMat.rows}, ${outMat.cols})")
    // println(s"debug: Wx_plus_B(${Wx_plus_B.rows}, ${Wx_plus_B.cols})")

    val batches: Array[DMD] = (
      for (i <- 0 until x.size) yield {
        Wx_plus_B(
          ::,
          i * out_width * out_width until i * out_width * out_width + out_width * out_width)
      }).toArray

    // println(s"debug: batches(${batches.size}, ${batches(0).rows}, ${batches(0).cols})")

    batches.map(_.t.toDenseVector)
  }

  def backward(d: DVD): DVD = {

    val dmap: ADVD = utils.divideIntoN(d, N = filter_set)
    val dMat: DMD = dmap.map(_.toDenseMatrix).reduce(DenseMatrix.vertcat(_, _))

    dB += sum(dMat, Axis._1)

    val col_X = this.col_X.get
    dW += dMat * col_X.t

    val col_W = this.col_W.get
    val x_shape = Shape(1, channel, input_width, input_width)
    val dxMat = col_W.t * dMat
    val dx = col2im(dxMat, x_shape, filter_width, filter_width, stride)
    val dxVec = dx.map(c => c.map(_.t.toDenseVector).reduce(DenseVector.vertcat(_, _))).reduce(DenseVector.vertcat(_, _))
    dxVec
  }

  override def backwards(d: ADVD): ADVD = {

    val dmap: Array[ADVD] = d.map(b => utils.divideIntoN(b, N = filter_set))
    val dMat: DMD = dmap.map(_.map(_.toDenseMatrix).reduce(DenseMatrix.vertcat(_, _))).reduce(DenseMatrix.horzcat(_, _))

    dB += sum(dMat, Axis._1)

    val col_X = this.col_X.get
    dW += dMat * col_X.t

    val col_W = this.col_W.get
    val x_shape = Shape(d.size, channel, input_width, input_width)
    val dxMat = col_W.t * dMat
    val dx: Array[ADMD] = col2im(dxMat, x_shape, filter_width, filter_width, stride)
    val dxArrVec: ADVD = dx.map(c => c.map(_.t.toDenseVector).reduce(DenseVector.vertcat(_, _))) //.reduce(DenseVector.vertcat(_,_))
    dxArrVec
  }

  def update(): Unit = {
    val wf: Array[DMD] = opt_filter.update(Array(col_W.get), Array(dW))

    // println(s"debug: F(${F.size}, ${F(0).size}, ${F(0)(0).size})")
    // println(s"debug: wf(${wf.size}, ${wf(0).rows}, ${wf(0).cols})")

    for {
      i <- F.indices
      j <- F(i).indices
    } {
      F(i)(j) -= wf(0)(i, j * F(i)(j).length until j * F(i)(j).length + F(i)(j).length).t
    }

    val wb: Array[DVD] = opt_bias.update(Array(B), Array(dB))
    B -= wb(0)

    reset()
  }

  def reset(): Unit = {
    col_X = None
    col_W = None
    dW = DenseMatrix.zeros[Double](filter_set, filter_width * filter_width * channel)
    dB = DenseVector.zeros[Double](dB.length)
  }

  def save(fn: String): Unit = {
    val fos = new java.io.FileOutputStream(fn, false)
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw = new java.io.PrintWriter(osw)

    val flatF = F.flatMap(_.map(_.toArray)).flatten
    pw.write(flatF.mkString(","))
    pw.write("\n")

    pw.write(B.toArray.mkString(","))
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
      fs <- 0 until B.length
    } {
      B(fs) = str(1)(fs)
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
      fs <- 0 until B.length
    } {
      B(fs) = blst(fs)
    }

    data.drop(2)
  }

  // helper

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

  def copy_B(): DVD = B.copy

  override def duplicate(): i2cConv = {
    val dup = i2cConv(input_width, filter_width, filter_set, channel, stride, distr, SD, update_method, lr)
    dup.F = this.copy_F()
    dup.B = this.copy_B()
    dup
  }

}

// test
object i2cConvTest {
  def main(args: Array[String]) {

    // Convolution(
    //    f: ActivateFunction,
    //    filter_width: Int, filter_set: Int = 1,
    //    channel: Int = 1, stride: Int = 1 )
    val fH = 3
    val fS = 2
    val ch = 3
    val stride = 2
    val input_width = 10
    val distr: String = "Xavier"
    val SD: Double = 0d
    val update_method: String = "AdaGrad"
    val lr: Double = 0.01
    val conv = i2cConv(input_width, fH, fS, ch, stride, distr, SD, update_method, lr)
    val img = DenseVector.range(0, ch * input_width * input_width).map(_.toDouble)
    val conv1 = conv.forward(img)
    println(conv1)
    println(s"length = ${conv1.size}")
    val back = conv.backward(conv1)
    println(s"img.size = ${img.size}")
    println(s"back.size = ${back.size}")
    conv.update()
    conv.reset()

    println("\n-----------\n")

    val conv2 = conv.forwards(Array(img, img, img))
    conv2.foreach(println)
    println(s"length = ${conv2.size}")
    val back2 = conv.backwards(conv2)
    println(s"img.size = ${img.size}")
    println(s"back2.size = ${back2.size}")
    conv.update()
    conv.reset()

    conv.save("tmp.csv")
    conv.load("tmp.csv")
    conv.save("tmp2.csv")
  }
}

