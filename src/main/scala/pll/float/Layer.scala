package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

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
  def get_value(src: scala.io.BufferedSource): T = {
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

  def forward(x: DenseVector[T]): DenseVector[T]

  def forwards(x: Array[DenseVector[T]]): Array[DenseVector[T]] = {
    // x.map(forward)
    (for (x_i <- x) yield { forward(x_i) }).toArray
  }

  def backward(d: DenseVector[T]): DenseVector[T]

  def backwards(d: Array[DenseVector[T]]): Array[DenseVector[T]] = {
    // d.map(backward)
    var buf = List[DenseVector[T]]()
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
  // def get_value(src:scala.io.BufferedSource): T = {
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

  def Xavier(row: Int, col: Int, node: Int): DenseMatrix[T] = {
    DenseMatrix.fill(row, col) {
      rand.nextGaussian * math.sqrt(1d / node)
    }
  }

  def Xavier(col: Int, node: Int): DenseVector[T] = {
    DenseVector.fill(col) {
      rand.nextGaussian * math.sqrt(1d / node)
    }
  }

  def He(row: Int, col: Int, node: Int): DenseMatrix[T] = {
    DenseMatrix.fill(row, col) {
      rand.nextGaussian * math.sqrt(2d / node)
    }
  }

  def He(col: Int, node: Int): DenseVector[T] = {
    DenseVector.fill(col) {
      rand.nextGaussian * math.sqrt(2d / node)
    }
  }

  def Gaussian(row: Int, col: Int, SD: T): DenseMatrix[T] = {
    DenseMatrix.fill(row, col) {
      rand.nextGaussian * SD
    }
  }

  def Gaussian(col: Int, SD: T): DenseVector[T] = {
    DenseVector.fill(col) {
      rand.nextGaussian * SD
    }
  }

  def Uniform(row: Int, col: Int, SD: T): DenseMatrix[T] = {
    DenseMatrix.fill(row, col) {
      rand.nextDouble * SD
    }
  }

  def Uniform(col: Int, SD: T): DenseVector[T] = {
    DenseVector.fill(col) {
      rand.nextDouble * SD
    }
  }

}
