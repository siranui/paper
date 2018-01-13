package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

object gradient {
  def numerical_diff(f: T => T, x: T) = {
    val h: T = (1e-8: T)

    var grad: T = (0: T)
    grad = (f(x + h) - f(x - h)) / (2 * h)

    grad
  }

  def numerical_gradient(f: DenseVector[T] => DenseVector[T], x: DenseVector[T]) = {
    val h: T = (1e-8: T)

    // var grad = x.map(_ => 0d)
    var grad = x *:* (0: T)
    grad = (f(x + h) - f(x - h)) / (2 * h)

    grad
  }
}
