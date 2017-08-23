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

  def divideIntoN(x: DenseVector[Double], N: Int): Array[DenseVector[Double]] = {
    val len = x.size / N
    (for (i <- 0 until N) yield {
      x(i * len until (i + 1) * len)
    }).toArray
  }

  def print_excuting_time(proc: => Unit): Unit = {
    val start = System.currentTimeMillis
    proc
    println((System.currentTimeMillis - start) + "msec")
  }

}