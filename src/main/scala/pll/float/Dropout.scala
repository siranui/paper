package pll.float

import breeze.linalg._
import typeAlias._

// dr = (0にする確率)
class Dropout(var dr: T) extends Layer {

  var masks: List[DenseVector[T]] = Nil

  def forward(x: DenseVector[T]): DenseVector[T] = {
    val mask = DenseVector.ones[T](x.size)
    for (i <- 0 until mask.size) {
      if (rand.nextDouble < dr) {
        mask(i) = (0: T)
      }
    }
    masks = mask :: masks

    x *:* mask
  }

  def forward_at_test(x: DenseVector[T]): DenseVector[T] = {
    x *:* ((1: T) - dr)
  }

  def backward(d: DenseVector[T]): DenseVector[T] = {
    val mask = masks.head
    masks = masks.tail

    d *:* mask
  }

  def backward_at_test(d: DenseVector[T]): DenseVector[T] = {
    d *:* ((1: T) - dr)
  }

  def update() {}
  def reset() { masks = Nil }
  def save(filename: String) {}
  def load(filename: String) {}
  override def load(data: List[String]) = { data }
}
