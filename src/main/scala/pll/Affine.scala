package pll


import breeze.linalg._

class Affine(
  val xn: Int,
  val yn: Int,
  val distr: String,
  val s: Double,
  val koshin: String,
  val a: Double) extends Layer {
  var opt: Opt = Opt.create(koshin, a)
  var w: DenseMatrix[Double] = DenseMatrix.zeros[Double](yn, xn)
  var b: DenseVector[Double] = DenseVector.zeros[Double](yn)
  var x1: List[DenseVector[Double]] = List[DenseVector[Double]]()
  if (distr == "Gaussian") {
    w = Gaussian(yn, xn, s)
    b = Gaussian(xn, s)
  }
  else if (distr == "Uniform") {
    w = Uniform(yn, xn, s)
    b = Uniform(xn, s)
  }
  else if (distr == "Xavier") {
    w = Xavier(yn, xn, xn)
    b = Xavier(yn, xn)
  }
  else {
    w = He(yn, xn, xn)
    b = He(yn, xn)
  }
  opt.register(Array(w))
  opt.register(Array(b))
  var wsum: DenseMatrix[Double] = DenseMatrix.zeros[Double](yn, xn)
  var bsum: DenseVector[Double] = DenseVector.zeros[Double](yn)

  def forward(x: DenseVector[Double]): DenseVector[Double] = {
    x1 = x :: x1
    w * x + b
  }

  def backward(d: DenseVector[Double]): DenseVector[Double] = {
    wsum += d * x1.head.t
    x1 = x1.tail
    bsum += d
    w.t * d
  }

  def update() {
    val tmp1 = opt.update(Array(w), Array(wsum))
    val tmp2 = opt.update(Array(b), Array(bsum))

    w = w - tmp1(0)
    b = b - tmp2(0)
  }

  def reset() {
    x1 = List[DenseVector[Double]]()
    wsum = DenseMatrix.zeros[Double](yn, xn)
    bsum = DenseVector.zeros[Double](yn)
  }

  def save(fn: String) {
    val fos = new java.io.FileOutputStream(fn, true)
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw = new java.io.PrintWriter(osw)
    for (i <- 0 until w.rows) {
      for (j <- 0 until w.cols) {
        pw.write(w(i, j).toString)
        if (i == w.rows - 1 && j == w.cols - 1) {
        }
        else {
          pw.write(",")
        }
      }
    }
    pw.write("\n")
    for (i <- 0 until b.size) {
      pw.write(b(i).toString)
      if (i != b.size - 1) {
        pw.write(",")
      }
    }
    pw.write("\n")
    pw.close()
  }

  def load(fn: String): Unit = {
    val str = io.Source.fromFile(fn).getLines.toArray.map(_.split(",").map(_.toDouble))
    for (i <- 0 until w.rows) {
      for (j <- 0 until w.cols) {
        w(i, j) = str(0)(w.cols * i + j)
      }
    }
    for (i <- 0 until b.size) {
      b(i) = str(1)(i)
    }
  }

  def load(data: List[String]): List[String] = {
    val ws = data(0).split(",").map(_.toDouble)
    val bs = data(1).split(",").map(_.toDouble)
    for (i <- 0 until w.rows) {
      for (j <- 0 until w.cols) {
        w(i, j) = ws(w.cols * i + j)
      }
    }
    for (i <- 0 until b.size) {
      b(i) = bs(i)
    }

    data.drop(2)
  }

  override def duplicate(): Affine = {
    val dup = new Affine(xn, yn, distr, s, koshin, a)
    dup.w = w.copy
    dup.b = b.copy
    dup
  }
}