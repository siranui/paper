package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

object utils {

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

  // for convolution
  def out_width(in_w: Int, fil_w: Int, stride: Int): Int = {
    (math.floor((in_w - fil_w) / stride) + 1).toInt
  }

  def divideIntoN[U](x: DenseVector[U], N: Int): Array[DenseVector[U]] = {
    val len = x.size / N
    (0 until N).map(i => x(i * len until (i + 1) * len)).toArray
  }

  /** read data from file.
    *
    * @param fn   filename. (relative path from root)
    * @param ds   read line size from head. if (ds == 0) then read all.   *
    * @param divV divide value.
    *
    * @return read data that divide divV( [0,255] -> [0,1) ).
    */
  def read(fn: String, ds: Int = 0, divV: T = 256)(
      implicit sepatate: String = ","): Array[DenseVector[T]] = {
    val f: Array[Array[T]] = ds match {
      case 0 =>
        io.Source
          .fromFile(fn)
          .getLines
          .map(_.split(sepatate).map(i => (i.toDouble: T) / divV).toArray)
          .toArray
      case _ =>
        io.Source
          .fromFile(fn)
          .getLines
          .take(ds)
          .map(_.split(sepatate).map(i => (i.toDouble: T) / divV).toArray)
          .toArray
    }
    val g = f.map(a => DenseVector(a))
    g
  }

  /** write data to file.
    *
    * @param fn       filename. (relative path from root)
    * @param dataList data.
    */
  def write[U](fn: String, dataList: Seq[DenseVector[U]], tf: Boolean = false): Unit = {
    val fos = new java.io.FileOutputStream(fn, tf) //true: 追記, false: 上書き
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw  = new java.io.PrintWriter(osw)

    for (data <- dataList.toList) {
      pw.write(data.activeValuesIterator.mkString(","))
      pw.write("\n")
    }
    pw.close()
  }

  def printExcutingTime[U](proc: => U): U = {
    val start     = System.currentTimeMillis
    val result: U = proc
    println(s"Excuting Time: ${System.currentTimeMillis - start} msec")

    result
  }

  def oneHot(x: Int, sz: Int = 10): DenseVector[T] = {
    assert(sz >= 1)
    assert(x >= 0 && x < sz, s"$x is out of the range( [0, ${sz - 1}] ).")

    DenseVector.tabulate(sz) { i =>
      if (i == x) 1d else 0d
    }
  }

  def oneHot2(x: Int, sz: Int = 10): SparseVector[T] = {
    assert(sz >= 1)
    assert(x >= 0 && x < sz, s"out of the range( [0, ${sz - 1}] ).")

    SparseVector(sz)((x, 1d))
  }

}
