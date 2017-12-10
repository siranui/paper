package fontGLO

import pll._
import breeze.linalg._

object img {
  def half(v: DenseVector[Double]) = {
    val wid = math.sqrt(v.size).toInt
    val c   = Convolution(wid, 2, stride = 2)
    c.F = Array(Array(DenseVector(0.25, 0.25, 0.25, 0.25)))
    c.forward(v)
  }

  def double(v: DenseVector[Double]) = {
    val wid = math.sqrt(v.size).toInt
    val mat = DenseMatrix.zeros[Double](wid * 2, wid * 2)
    val tmp = reshape(v, wid, wid).t
    for (i <- 0 until wid; j <- 0 until wid) {
      mat(i * 2 until i * 2 + 2, j * 2 until j * 2 + 2) += tmp(i, j)
    }
    mat.t.toDenseVector
  }
}
