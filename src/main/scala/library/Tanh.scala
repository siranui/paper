package pll
import breeze.linalg._

class Tanh() extends Layer{
  var y:DenseVector[Double]=null
  def forward(x:DenseVector[Double])={
    y=x.map(a=>(math.exp(a)-math.exp(-a))/(math.exp(a)+math.exp(-a)))
    y
  }

  def backward(d:DenseVector[Double])={
    val d1=d*:*(1d-y*:*y)
    d1
  }
  def update(){
  }
  def reset(){
  }
  def save(fn:String){
  }
  def load(fn:String){
  }
  def load(data: List[String]) = {
    data
  }
  override def duplicate()={
    new Tanh()
  }
}
