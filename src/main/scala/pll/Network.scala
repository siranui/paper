package pll


import breeze.linalg._

class Network() {
  val rand = new util.Random(0)

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

  def train(
    inputs: Seq[DenseVector[Double]],
    tags: Seq[DenseVector[Double]],
    calcError: (DenseVector[Double], DenseVector[Double]) => Double,
    calcGrad: (DenseVector[Double], DenseVector[Double]) => DenseVector[Double],
  ) {
    var E = 0d
    for((x, t) <- (inputs zip tags)){
      val y = predict(x)
      val d = calcGrad(y, t)
      E += calcError(y, t)
      update(d)
    }
    println(s"E:$E")
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

  def batch_train(
    inputs: Seq[DenseVector[Double]],
    tags: Seq[DenseVector[Double]],
    batchSize: Int,
    calcError: (ADV, ADV) => Double,
    calcGrad: (ADV, ADV) => ADV
  ) {
    var E = 0d
    var unusedIdx = rand.shuffle(List.range(0, inputs.size))
    while (unusedIdx.nonEmpty) {
      val batchMask = unusedIdx.take(batchSize)
      unusedIdx = unusedIdx.drop(batchSize)

      val xs = batchMask.map(idx => inputs(idx)).toArray
      val ts = batchMask.map(idx => tags(idx)).toArray
      val ys = predict(xs)

      E += calcError(ys, ts)
      val d = calcGrad(ys, ts)

      update(d)
    }
    println(s"E:$E")
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
