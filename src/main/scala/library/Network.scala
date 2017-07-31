package pll

import breeze.linalg._

class Network() {
  var layers = List[Layer]()

  def add(layer: Layer) = {
    layers = (layer :: layers.reverse).reverse
    this
  }

  def predict(x: DenseVector[Double]) = {
    var tmp = x
    for(layer <- layers){
      tmp = layer.forward(tmp)
    }
    tmp
  }

  def calc_L2(y: DenseVector[Double], t:DenseVector[Double])={
    val E = y -:- t
    sum((E *:* E) /:/ 2d)
  }

  def calc_cross_entropy_loss(y: DenseVector[Double], t:DenseVector[Double]) = {
    - sum(t *:* breeze.numerics.log(y))
  }

  def calc_L2_grad(y: DenseVector[Double], t:DenseVector[Double])={
    y -:- t
  }

  def backprop(d: DenseVector[Double]) = {
    var tmp = d
    val rLayers = layers.reverse
    for(rLayer <- rLayers){
      tmp = rLayer.backward(tmp)
    }
    tmp
  }

  def update(){
    layers.map(_.update)
  }

  def reset() { layers.map(_.reset) }

  def update(d: DenseVector[Double]) {
    backprop(d)
    update()
    reset()
  }

  def save(fn: String) {
    layers.foreach(_.save(fn))
  }

  def load(fn: String) {
    var tmp = io.Source.fromFile(fn).getLines.toList
    for(l <- layers){
      tmp = l.load(tmp)
    }
  }
}

class NetworkWithDropout() extends Network{
  def forward_at_test(x: DenseVector[Double]) = {
    var tmp = x
    for(layer <- layers){
      if(layer.isInstanceOf[Dropout]){
        tmp = layer.asInstanceOf[Dropout].forward_at_test(tmp)
      } else {
        tmp = layer.forward(tmp)
      }
    }
    tmp
  }
}

class batchNet() extends Network {
  type ADV = Array[DenseVector[Double]]

  def predict(xs: ADV): ADV = {
    var tmp = xs
    for(layer <- layers){
      tmp = layer.forwards(tmp)
    }
    tmp
  }

  def calc_L2(ys: ADV, ts: ADV): Double = {
    var E = 0d
    for((y,t) <- ys zip ts){
      val tmp = y -:- t
      E += sum((tmp *:* tmp) /:/ 2d)
    }
    E
  }

  def calc_cross_entropy_loss(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Double = {
    var L = 0d
    for((y,t) <- ys zip ts){
      L += - sum(t *:* breeze.numerics.log(y))
    }
    L
  }

  def calc_L2_grad(ys: Array[DenseVector[Double]], ts:Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val grads = for((y,t) <- ys zip ts) yield { y -:- t }
    grads.toArray
  }

  def backprop(ds: ADV) = {
    var tmp = ds.reverse
    val rLayers = layers.reverse
    for(rLayer <- rLayers){
      tmp = rLayer.backwards(tmp)
    }
    tmp.reverse
  }

  def update(ds: ADV) {
    backprop(ds)
    update()
    reset()
  }
}
