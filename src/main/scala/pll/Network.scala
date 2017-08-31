package pll


import breeze.linalg._

class Network() {
  var layers = List[Layer]()

  def add(layer: Layer): Network = {
    layers = (layer :: layers.reverse).reverse
    this
  }

  def predict(x: DenseVector[Double]): DenseVector[Double] = {
    var predict_value = x
    for (layer <- layers) {
      predict_value = layer.forward(predict_value)
    }
    predict_value
  }

  def calc_L2(y: DenseVector[Double], t: DenseVector[Double]): Double = {
    val E = y -:- t
    sum((E *:* E) /:/ 2d)
  }

  def calc_cross_entropy_loss(y: DenseVector[Double], t: DenseVector[Double]): Double = {
    -sum(t *:* breeze.numerics.log(y))
  }

  def calc_L2_grad(y: DenseVector[Double], t: DenseVector[Double]): DenseVector[Double] = {
    y -:- t
  }

  def backprop(d: DenseVector[Double]): DenseVector[Double] = {
    var tmp = d
    val rLayers = layers.reverse
    for (rLayer <- rLayers) {
      tmp = rLayer.backward(tmp)
    }
    tmp
  }

  def update() {
    layers.foreach(_.update())
  }

  def reset() {
    layers.foreach(_.reset())
  }

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
    for (l <- layers) {
      tmp = l.load(tmp)
    }
  }
}

class NetworkWithDropout() extends Network {
  def forward_at_test(x: DenseVector[Double]): DenseVector[Double] = {
    var forward_value = x
    for (layer <- layers) {
      forward_value = layer match {
        case dropout: Dropout => dropout.forward_at_test(forward_value)
        case _: Layer         => layer.forward(forward_value)
      }
    }
    forward_value
  }
}

class batchNet() extends Network {
  type ADV = Array[DenseVector[Double]]

  def predict(xs: ADV): ADV = {
    var tmp = xs
    for (layer <- layers) {
      tmp = layer.forwards(tmp)
    }
    tmp
  }

  def calc_L2(ys: ADV, ts: ADV): Double = {
    var E = 0d
    for ((y, t) <- ys zip ts) {
      val tmp = y -:- t
      E += sum((tmp *:* tmp) /:/ 2d)
    }
    E
  }

  def calc_cross_entropy_loss(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Double = {
    var L = 0d
    for ((y, t) <- ys zip ts) {
      L += -sum(t *:* breeze.numerics.log(y))
    }
    L
  }

  def calc_L2_grad(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val grads = for ((y, t) <- ys zip ts) yield {
      y -:- t
    }
    grads
  }

  def backprop(ds: ADV): Array[DenseVector[Double]] = {
    var tmp = ds.reverse
    val rLayers = layers.reverse
    for (rLayer <- rLayers) {
      tmp = rLayer.backwards(tmp)
    }
    tmp.reverse
  }

  def update(ds: ADV): Unit = {
    backprop(ds)
    update()
    reset()
  }
}
