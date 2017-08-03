package pll
import breeze.linalg._

class Dropout(var dr: Double) extends Layer {

  var masks: List[DenseVector[Double]] = Nil

  def forward(x: DenseVector[Double]) = {
    val mask = DenseVector.ones[Double](x.size)
    for(i <- 0 until mask.size){
      if(rand.nextDouble < dr){
        mask(i) = 0d
      }
    }
    masks = mask :: masks

    x *:* mask
  }

  def forward_at_test(x: DenseVector[Double]) = {
    x *:* (1d - dr)
  }

  def backward(d: DenseVector[Double]) = {
    val mask = masks.head
    masks = masks.tail

    d *:* mask
  }

  def backward_at_test(d: DenseVector[Double]) = {
    d *:* (1d - dr)
  }

  def update() {}
  def reset() {masks = Nil}
  def save(filename: String) {}
  def load(filename: String) {}
  def load(data: List[String]) = { data }
}
