package pll

import breeze.linalg._

object grad {
  def numerical_diff(f: Double => Double, x: Double) = {
    val h = 1e-8

    var grad = 0d
    grad = (f(x+h) - f(x-h)) / (2*h)

    grad
  }

  def numerical_gradient(f: DenseVector[Double] => DenseVector[Double], x: DenseVector[Double]) = {
    val h = 1e-8

    var grad = x.map(_=>0d)
    grad = (f(x+h) - f(x-h)) / (2*h)

    grad
  }
}
