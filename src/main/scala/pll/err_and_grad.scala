package pll


import breeze.linalg._

object err {
  type ADV = Array[DenseVector[Double]]

  def calc_L2(y: DenseVector[Double], t: DenseVector[Double]): Double = {
    val E = y -:- t
    sum((E *:* E) /:/ 2d)
  }

  def calc_L2(ys: ADV, ts: ADV): Double = {
    var E = 0d
    for ((y, t) <- ys zip ts) {
      val tmp = y -:- t
      E += sum((tmp *:* tmp) /:/ 2d)
    }
    E
  }

  def calc_cross_entropy_loss(y: DenseVector[Double], t: DenseVector[Double]): Double = {
    -sum(t *:* breeze.numerics.log(y))
  }

  def calc_cross_entropy_loss(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Double = {
    var L = 0d
    for ((y, t) <- ys zip ts) {
      L += -sum(t *:* breeze.numerics.log(y))
    }
    L
  }

}

object grad {
  def calc_L2_grad(y: DenseVector[Double], t: DenseVector[Double]): DenseVector[Double] = {
    y -:- t
  }

  def calc_L2_grad(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val grads = for ((y, t) <- ys zip ts) yield {
      y -:- t
    }
    grads
  }

  def calc_cross_entropy_grad(y: DenseVector[Double], t: DenseVector[Double]): DenseVector[Double] = {
    y -:- t
  }

  def calc_cross_entropy_grad(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val grads = for ((y, t) <- ys zip ts) yield {
      y -:- t
    }
    grads
  }

}
