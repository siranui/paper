package pll

import breeze.linalg._

case class ResNet(L: Seq[Layer]) extends Layer {

  type T  = Double
  type DV = DenseVector[T]
  type DM = DenseMatrix[T]

  def forward(x: DV): DV = {
    var tmp = x
    for (l <- L) {
      tmp = l.forward(tmp)
    }
    tmp + x
  }

  def backward(d: DV): DV = {
    var tmp = d
    for (l <- L) {
      tmp = l.forward(tmp)
    }
    tmp + d
  }

  def save(filename: String): Unit = {}

  def load(filename: String): Unit = {}

  def load(data: List[String]): List[String] = data

  def reset(): Unit = { L.map(_.reset()) }

  def update(): Unit = { L.map(_.update()) }

}
