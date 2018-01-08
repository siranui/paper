package pll

import breeze.linalg._

trait Load {
  def load(filename: String): Unit

  def load(data: List[String]): List[String] = {
    println(
      s"WARNING[load(data: List[String]) in ${this.getClass.getName}]: You should implement this function")
    data
  }

  def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {
    println(
      s"WARNING[load_version_iterator(data_iter: scala.io.BufferedSource) in ${this.getClass.getName}]: You should implement this function")
  }

  // src = getLinesする前の値
  def get_value(src: scala.io.BufferedSource): Double = {
    val builder = new StringBuilder
    var c       = src.next
    while (c != '\n' && c != ',') {
      builder.append(c)
      c = src.next
    }

    if (builder.length == 0) get_value(src) else builder.result.toDouble
  }

}

trait Layer extends Load {

  def forward(x: DenseVector[Double]): DenseVector[Double]

  def forwards(x: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    // x.map(forward)
    (for (x_i <- x) yield { forward(x_i) }).toArray
  }

  def backward(d: DenseVector[Double]): DenseVector[Double]

  def backwards(d: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    // d.map(backward)
    var buf = List[DenseVector[Double]]()
    for (d_i <- d.reverse) {
      buf = backward(d_i) :: buf
    }
    buf.toArray
  }

  def update(): Unit

  def reset(): Unit

  def duplicate(): Layer = {
    this
  }

  def save(filename: String): Unit

  def save_(pw: java.io.PrintWriter): java.io.PrintWriter = {
    println(
      s"WARNING[save(pw: java.io.PrintWriter) in ${this.getClass.getName}]: You should implement this function")
    pw
  }

  // // src = getLinesする前の値
  // def get_value(src:scala.io.BufferedSource): Double = {
  //   val builder = new StringBuilder
  //   var c = src.next
  //   while(c != '\n' && c != ',') {
  //     builder.append(c)
  //     c = src.next
  //   }

  //   if( builder.length == 0 ) get_value(src) else builder.result.toDouble
  // }

  // def load(filename: String): Unit

  // def load(data: List[String]): List[String] = {
  //   println(s"WARNING[load(data: List[String]) in ${this.getClass.getName}]: You should implement this function")
  //   data
  // }

  // def load_version_iterator(data_iter: scala.io.BufferedSource): Unit = {
  //   println(s"WARNING[load_version_iterator(data_iter: scala.io.BufferedSource) in ${this.getClass.getName}]: You should implement this function")
  // }

  val rand = new util.Random(0)

  def Xavier(row: Int, col: Int, node: Int): DenseMatrix[Double] = {
    DenseMatrix.fill(row, col) {
      rand.nextGaussian * math.sqrt(1d / node)
    }
  }

  def Xavier(col: Int, node: Int): DenseVector[Double] = {
    DenseVector.fill(col) {
      rand.nextGaussian * math.sqrt(1d / node)
    }
  }

  def He(row: Int, col: Int, node: Int): DenseMatrix[Double] = {
    DenseMatrix.fill(row, col) {
      rand.nextGaussian * math.sqrt(2d / node)
    }
  }

  def He(col: Int, node: Int): DenseVector[Double] = {
    DenseVector.fill(col) {
      rand.nextGaussian * math.sqrt(2d / node)
    }
  }

  def Gaussian(row: Int, col: Int, SD: Double): DenseMatrix[Double] = {
    DenseMatrix.fill(row, col) {
      rand.nextGaussian * SD
    }
  }

  def Gaussian(col: Int, SD: Double): DenseVector[Double] = {
    DenseVector.fill(col) {
      rand.nextGaussian * SD
    }
  }

  def Uniform(row: Int, col: Int, SD: Double): DenseMatrix[Double] = {
    DenseMatrix.fill(row, col) {
      rand.nextDouble * SD
    }
  }

  def Uniform(col: Int, SD: Double): DenseVector[Double] = {
    DenseVector.fill(col) {
      rand.nextDouble * SD
    }
  }

}
