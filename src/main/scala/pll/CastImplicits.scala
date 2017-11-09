package pll


object CastImplicits {
  import scala.language.implicitConversions._
  import breeze.linalg._

  implicit def Double2Float(n: Double) = n.toFloat
  implicit def Double2Float(n: DenseVector[Double]) = convert(n, Float)
  implicit def Double2Float(n: DenseMatrix[Double]) = convert(n, Float)
}
