package pll.float

object typeAlias {
  import breeze.linalg._
  type T       = Float
  type DV      = DenseVector[T]
  type DM      = DenseMatrix[T]
  type DATASET = Array[(DV, DV)]
}
