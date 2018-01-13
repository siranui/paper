package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

case class BNL(var xn: Int, var bn: Int) extends Layer {
  var beta  = DenseVector.zeros[T](xn)
  var dbeta = DenseVector.zeros[T](xn)
  var gamma = DenseVector.ones[T](xn)
  gamma = Gaussian(xn, (0.01: T)).map(_ + (1: T))
  var dgamma   = DenseVector.zeros[T](xn)
  val eps: T   = 1e-5
  var xhat     = new Array[DenseVector[T]](0)
  var xmu      = new Array[DenseVector[T]](0)
  var ivar     = DenseVector[T]()
  var sqrtvar  = DenseVector[T]()
  var varia    = DenseVector[T]()
  var opt      = Opt.create("SGD", (0.01: T))
  var allvaria = DenseVector.zeros[T](xn)
  var allmu    = DenseVector.zeros[T](xn)

  override def forwards(x: Array[DenseVector[T]]): Array[DenseVector[T]] = {
    var sum1 = DenseVector.zeros[T](x(0).size)
    for (i <- 0 until x.size) {
      sum1 = sum1 +:+ x(i)
    }
    val mu = ((1: T) / x.size) *:* sum1

    xmu = x.map(_ - mu)

    val sq = xmu.map { case a => a * a }

    var sum2 = DenseVector.zeros[T](x(0).size)
    for (i <- 0 until x.size) {
      sum2 = sum2 +:+ sq(i)
    }
    varia = ((1: T) / x.size) *:* sum2

    sqrtvar = varia.map { case a => math.sqrt(a + eps) }

    ivar = sqrtvar.map { case a => (1: T) / a }

    xhat = xmu.map(_ *:* ivar)

    val gammax = xhat.map(_ *:* gamma)

    val out = gammax.map(_ +:+ beta)

    allmu = (0.9: T) * allmu +:+ (0.1: T) * mu
    allvaria = (0.9: T) * allvaria +:+ (0.1: T) * varia

    out
  }

  def forward(x: DenseVector[T]) = {
    val allmux   = x -:- allmu
    val xhattest = allmux /:/ (allvaria.map { case a => (math.sqrt(a + eps): T) })
    val gammax   = xhattest *:* gamma
    val out      = gammax +:+ beta
    out
  }

  def backward(d: DenseVector[T]) = { d }

  override def backwards(dout: Array[DenseVector[T]]): Array[DenseVector[T]] = {
    for (i <- 0 until dout.size) {
      dbeta = dbeta +:+ dout(i)
    }
    val dgammax = dout

    for (i <- 0 until dout.size) {
      dgamma += dgammax(i) *:* xhat(i)
    }
    val dxhat = dgammax.map(_ *:* gamma)

    val divar = DenseVector.zeros[T](dout(0).size)
    for (i <- 0 until dout.size) {
      divar += dxhat(i) *:* xmu(i)
    }
    val dxmu1 = dxhat.map(_ *:* ivar)

    val dsqrtvar = ((-1: T) / (sqrtvar *:* sqrtvar)) *:* divar

    val dvar: DenseVector[T] = (0.5: T) *:* ((1: T) / (varia +:+ eps).map(v => (math.sqrt(v): T))) *:* dsqrtvar

    val vec1 = (0 until dout.size).toArray.map { case a => dvar }
    val dsq  = vec1.map(_ *:* ((1: T) / dout.size))

    val dxmu2 = xmu.zip(dsq).map { case (a, b) => (2: T) *:* a *:* b }

    val dx1 = dxmu1.zip(dxmu2).map { case (a, b) => a +:+ b }
    val dmu = DenseVector.zeros[T](dout(0).size)
    for (i <- 0 until dout.size) {
      dmu -= dx1(i)
    }

    val vec2 = (0 until dout.size).toArray.map { case a => dmu }
    val dx2  = vec2.map(_ *:* ((1: T) / dout.size))

    val dx = dx1.zip(dx2).map { case (a, b) => a +:+ b }

    dx
  }

  def update() {
    val tmp1 = opt.update(Array(gamma), Array(dgamma))
    val tmp2 = opt.update(Array(beta), Array(dbeta))

    gamma = gamma - (tmp1(0))
    beta = beta - (tmp2(0))
    reset()
  }

  def reset() {
    dgamma = DenseVector.zeros[T](xn)
    dbeta = DenseVector.zeros[T](xn)
  }
  // def allsumreset(){
  //   var allsum1 = DenseVector.zeros[T](xn)
  //   var allsum2 = DenseVector.zeros[T](xn)
  //   var count = 0
  // }

  def save(fn: String) {
    val fos = new java.io.FileOutputStream(fn, false)
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw  = new java.io.PrintWriter(osw)
    for (i <- 0 until gamma.size) {
      pw.write(gamma(i).toString)
      if (i != gamma.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    for (i <- 0 until beta.size) {
      pw.write(beta(i).toString)
      if (i != beta.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")

    pw.write(allvaria.toArray.mkString(","))
    pw.write("\n")

    pw.write(allmu.toArray.mkString(","))
    pw.write("\n")

    pw.close()

    opt.save(fn)
  }

  override def save_(pw: java.io.PrintWriter) = {

    pw.write(gamma.toArray.mkString(","))
    pw.write("\n")

    pw.write(beta.toArray.mkString(","))
    pw.write("\n")

    pw.write(allvaria.toArray.mkString(","))
    pw.write("\n")

    pw.write(allmu.toArray.mkString(","))
    pw.write("\n")

    opt.save_(pw)

    pw
  }

  def load(fn: String) {
    val str = io.Source.fromFile(fn).getLines.toArray.map(_.split(",").map(_.toDouble))
    println(s"BN-load:")
    println(s"\tgamma: ${gamma.size}, loaded: ${str(0).size}")
    println(s"\tbeta: ${beta.size}, loaded: ${str(1).size}")
    println(s"\tallvaria: ${allvaria.size}, loaded: ${str(2).size}")
    println(s"\tallmu: ${allmu.size}, loaded: ${str(3).size}")
    for (i <- 0 until gamma.size) {
      gamma(i) = str(0)(i)
    }
    for (i <- 0 until beta.size) {
      beta(i) = str(1)(i)
    }
    for (i <- 0 until allvaria.size) {
      allvaria(i) = str(2)(i)
    }
    for (i <- 0 until allmu.size) {
      allmu(i) = str(3)(i)
    }

    opt.load(fn)
  }

  override def load(data: List[String]): List[String] = {
    val str = data.take(4).map(_.split(",").map(_.toDouble))

    pll.log.debug(s"BN-load:")
    pll.log.debug(s"\tgamma: ${gamma.size}, loaded: ${str(0).size}")
    pll.log.debug(s"\tbeta: ${beta.size}, loaded: ${str(1).size}")
    pll.log.debug(s"\tallvaria: ${allvaria.size}, loaded: ${str(2).size}")
    pll.log.debug(s"\tallmu: ${allmu.size}, loaded: ${str(3).size}")

    for (i <- 0 until gamma.size) {
      gamma(i) = str(0)(i)
    }
    for (i <- 0 until beta.size) {
      beta(i) = str(1)(i)
    }
    for (i <- 0 until allvaria.size) {
      allvaria(i) = str(2)(i)
    }
    for (i <- 0 until allmu.size) {
      allmu(i) = str(3)(i)
    }

    opt.load(data.drop(4))
  }

  override def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {

    for (i <- 0 until gamma.size) {
      gamma(i) = get_value(data_iter)
    }
    for (i <- 0 until beta.size) {
      beta(i) = get_value(data_iter)
    }
    for (i <- 0 until allvaria.size) {
      allvaria(i) = get_value(data_iter)
    }
    for (i <- 0 until allmu.size) {
      allmu(i) = get_value(data_iter)
    }

    opt.load_version_iterator(data_iter)
  }
}
