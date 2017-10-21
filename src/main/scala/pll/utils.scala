package pll


import breeze.linalg._

object utils {

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

  // for convolution
  def out_width(in_w: Int, fil_w: Int, stride: Int): Int = {
    (math.floor((in_w - fil_w) / stride) + 1).toInt
  }

  def divideIntoN[T](x: DenseVector[T], N: Int): Array[DenseVector[T]] = {
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
  def read(fn: String, ds: Int = 0, divV: Double = 256): Array[DenseVector[Double]] = {
    val f = ds match {
      case 0 =>
        io.Source.fromFile(fn).getLines
          .map(_.split(",").map(_.toDouble / divV).toArray).toArray
      case _ =>
        io.Source.fromFile(fn).getLines.take(ds)
          .map(_.split(",").map(_.toDouble / divV).toArray).toArray
    }
    val g = f.map(a => DenseVector(a))
    g
  }

  /** write data to file.
   *
   * @param fn       filename. (relative path from root)
   * @param dataList data.
   */
  def write(fn: String, dataList: List[DenseVector[Int]], tf: Boolean = false): Unit = {
    val fos = new java.io.FileOutputStream(fn, tf) //true: è¿½è¨, false: ä¸æ¸ã
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw = new java.io.PrintWriter(osw)

    for (data <- dataList) {
      pw.write(data.toArray.mkString(","))
      pw.write("\n")
    }
    pw.close()
  }

  def printExcutingTime[T](proc: => T): T = {
    val start = System.currentTimeMillis
    val result: T = proc
    println(s"Excuting Time: ${System.currentTimeMillis - start} msec")

    result
  }


  def oneHot(x: Int, sz: Int = 10) = {
    assert(sz >= 1)
    assert(x >= 0 && x < sz, s"out of the range( [0, ${sz-1}] ).")

    DenseVector.tabulate(sz){i => if(i == x) 1d else 0d}
  }


}
