package pll
import breeze.linalg._

class Sigmoid() extends Layer{

  var y:DenseVector[Double] = null
  def forward(x:DenseVector[Double])= {
    y = x.map(a => 1/(1+math.exp(-a)))
    y
  }
  def backward(d:DenseVector[Double])= {
    y *:* (1d - y)
  }
  def update(){
  }
  def reset(){
  }
  def save(fn:String){
  }
  def load(data: List[String]) = {
    data
  }
  override def duplicate()={
    new Sigmoid()
  }
}
