package pll

object typeAlias {
  import breeze.linalg._
  type T       = Double
  type DV      = DenseVector[T]
  type DM      = DenseMatrix[T]
  type DATASET = Array[(DV, DV)]
}
