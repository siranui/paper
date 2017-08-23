package pll


import breeze.linalg._

trait Layer {

  def forward(x: DenseVector[Double]): DenseVector[Double]

  def forwards(x: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    x.map(forward)
  }

  def backward(d: DenseVector[Double]): DenseVector[Double]

  def backwards(d: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    d.map(backward)
  }

  def update(): Unit

  def reset(): Unit

  def duplicate(): Layer = {
    this
  }

  def save(filename: String): Unit

  def load(filename: String): Unit

  def load(data: List[String]): List[String]

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
