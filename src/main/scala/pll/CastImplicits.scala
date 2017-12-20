package pll

object CastImplicits {
  import breeze.linalg._

  implicit def Double2Float(n: Double)              = n.toFloat
  implicit def Double2Float(n: DenseVector[Double]) = convert(n, Float)
  implicit def Double2Float(n: DenseMatrix[Double]) = convert(n, Float)
}

// object Types {
//   import breeze.linalg._
//
//   type T = Float
//   type DV = DenseVector[T]
//   type DM = DenseMatrix[T]
// }
