package pll.float

import breeze.linalg._
import typeAlias._
import CastImplicits._

class Network() {
  val rand = new util.Random(0)

  var layers = List[Layer]()

  def add(layer: Layer): Network = {
    layers = (layer :: layers.reverse).reverse
    this
  }

  def predict(x: DenseVector[T]): DenseVector[T] = {
    var predict_value = x
    for (layer <- layers) {
      predict_value = layer.forward(predict_value)
    }
    predict_value
  }

  def forward_at_test(x: DenseVector[T]): DenseVector[T] = {
    var forward_value = x
    for (layer <- layers) {
      forward_value = layer match {
        case dropout: Dropout => dropout.forward_at_test(forward_value)
        case _: Layer         => layer.forward(forward_value)
      }
    }
    forward_value
  }

  def train(
      inputs: Seq[DenseVector[T]],
      tags: Seq[DenseVector[T]],
      calcError: (DenseVector[T], DenseVector[T]) => T,
      calcGrad: (DenseVector[T], DenseVector[T]) => DenseVector[T],
  ): (T, List[DenseVector[T]]) = {
    var E: T                     = 0
    var ys: List[DenseVector[T]] = Nil
    for ((x, t) <- (inputs zip tags)) {
      val y = predict(x)
      val d = calcGrad(y, t)
      E += calcError(y, t)
      ys = y :: ys
      update(d)
    }
    (E, ys.reverse)
  }

  def test(
      inputs: Seq[DenseVector[T]],
      tags: Seq[DenseVector[T]],
      calcError: (DenseVector[T], DenseVector[T]) => T,
  ): (T, List[DenseVector[T]]) = {
    var E: T                     = 0
    var ys: List[DenseVector[T]] = Nil
    for ((x, t) <- (inputs zip tags)) {
      // val y = predict(x)
      val y = forward_at_test(x)
      reset()
      E += calcError(y, t)
      ys = y :: ys
    }
    (E, ys.reverse)
  }

  def backprop(d: DenseVector[T]): DenseVector[T] = {
    var tmp     = d
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

  def update(d: DenseVector[T]) {
    backprop(d)
    update()
    reset()
  }

  def save(fn: String) {
    layers.foreach(_.save(fn))
  }

  def save_one_file(fn: String) {
    val fos = new java.io.FileOutputStream(fn, false)
    val osw = new java.io.OutputStreamWriter(fos, "UTF-8")
    val pw  = new java.io.PrintWriter(osw)
    layers.foreach(_.save_(pw))
    pw.close()
  }

  def load(fn: String) {
    var tmp = io.Source.fromFile(fn).getLines.toList
    pll.log.debug(s"$fn:L${tmp.size}")
    for (l <- layers) {
      tmp = l.load(tmp)
    }
  }

  // iterator version
  def load_version_iterator(fn: String) {
    val iter = io.Source.fromFile(fn)
    for (l <- layers) {
      l.load_version_iterator(iter)
    }

    if (iter.hasNext) pll.log.warn(s"$fn has next.")
  }
}

class NetworkWithDropout() extends Network {
  override def forward_at_test(x: DenseVector[T]): DenseVector[T] = {
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
  type ADV = Array[DenseVector[T]]

  def predict(xs: ADV): ADV = {
    var tmp = xs
    for (layer <- layers) {
      tmp = layer.forwards(tmp)
    }
    tmp
  }

  def batch_train(
      inputs: Seq[DenseVector[T]],
      tags: Seq[DenseVector[T]],
      batchSize: Int,
      calcError: (ADV, ADV) => T,
      calcGrad: (ADV, ADV) => ADV
  ) = {
    var E: T                             = 0
    var yslst: List[Seq[DenseVector[T]]] = Nil
    var unusedIdx                        = rand.shuffle(List.range(0, inputs.size))
    while (unusedIdx.nonEmpty) {
      val batchMask = unusedIdx.take(batchSize)
      unusedIdx = unusedIdx.drop(batchSize)

      val xs = batchMask.map(idx => inputs(idx)).toArray
      val ts = batchMask.map(idx => tags(idx)).toArray
      val ys = predict(xs)

      yslst = ys :: yslst
      E += calcError(ys, ts)
      val d = calcGrad(ys, ts)

      update(d)
    }

    (E, yslst.reverse)
  }

  def backprop(ds: ADV): Array[DenseVector[T]] = {
    // var tmp = ds.reverse
    var tmp     = ds
    val rLayers = layers.reverse
    for (rLayer <- rLayers) {
      tmp = rLayer.backwards(tmp)
    }
    // tmp.reverse
    // TODO: 順番をきちんと把握する
    tmp.reverse
  }

  def update(ds: ADV): Unit = {
    backprop(ds)
    update()
    reset()
  }
}
