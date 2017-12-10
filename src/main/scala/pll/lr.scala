package pll
import breeze.linalg._
trait Opt {
  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]];
  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]];
  def register(ps: Array[DenseMatrix[Double]]);
  def register(ps: Array[DenseVector[Double]]);
}
object Opt {
  def create(name: String, lr: Double) = {
    val sp = name.split(",")
    if (sp(0) == "Momentum") {
      if (sp.size < 2) {
        if (lr == 0d)
          new Momentum()
        else
          new Momentum(lr)
      }
      else
        new Momentum(lr, sp(1).toDouble)
    }
    else if (sp(0) == "AdaGrad") {
      //println("AdaGrad")
      if (sp.size < 2) {
        if (lr == 0d)
          new AdaGrad()
        else
          new AdaGrad(lr)
      }
      else
        new AdaGrad(lr, sp(1).toDouble)
    }
    else if (sp(0) == "Adam") {
      //println("Adam")
      if (sp.size < 4) {
        if (lr == 0d)
          new Adam()
        else
          new Adam(lr)
      }
      else
        new Adam(lr, sp(1).toDouble, sp(2).toDouble, sp(3).toDouble)
    }
    else if (sp(0) == "RMSProp") {
      //println("RMSProp")
      if (sp.size < 3) {
        if (lr == 0d)
          new RMSProp()
        else
          new RMSProp(lr)
      }
      else
        new RMSProp(lr, sp(1).toDouble, sp(2).toDouble)
    }
    else
      new SGD(lr)
  }
}
class SGD(var lr: Double = 0.01) extends Opt {
  def register(ps: Array[DenseMatrix[Double]]) {}
  def register(ps: Array[DenseVector[Double]]) {}

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    val us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      us(i) = lr * ds(i)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      us(i) = lr * ds(i)
    }
    us
  }
}

class Momentum(var lr: Double = 0.01, var momentum: Double = 0.9) extends Opt {
  var vms = Array[DenseMatrix[Double]]()
  var vvs = Array[DenseVector[Double]]()

  def register(ps: Array[DenseMatrix[Double]]) {
    vms = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until vms.size)
      vms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
  }

  def register(ps: Array[DenseVector[Double]]) {
    vvs = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until vvs.size)
      vvs(i) = DenseVector.zeros[Double](ps(i).size)
  }

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    val us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      vms(i) = (momentum * vms(i)) -:- (lr * ds(i))
      us(i) = -1d * vms(i)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      vvs(i) = (momentum * vvs(i)) -:- (lr * ds(i))
      us(i) = -1d * vvs(i)
    }
    us
  }
}

class AdaGrad(var lr: Double = 0.01, var epsilon: Double = 1e-8) extends Opt {
  var hms = Array[DenseMatrix[Double]]()
  var hvs = Array[DenseVector[Double]]()

  def register(ps: Array[DenseMatrix[Double]]) {
    hms = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until hms.size)
      hms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
  }

  def register(ps: Array[DenseVector[Double]]) {
    hvs = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until hvs.size)
      hvs(i) = DenseVector.zeros[Double](ps(i).size)
  }

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    val us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      hms(i) = hms(i) + ds(i) *:* ds(i)
      us(i) = lr * ds(i) /:/ (hms(i).map(Math.sqrt) + epsilon)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      hvs(i) = hvs(i) + ds(i) *:* ds(i)
      us(i) = lr * ds(i) /:/ (hvs(i).map(Math.sqrt) + epsilon)
    }
    us
  }
}

class Adam(val lr: Double = 0.001,
           val beta1: Double = 0.9,
           val beta2: Double = 0.999,
           val epsilon: Double = 1e-8)
    extends Opt {
  var vms = Array[DenseMatrix[Double]]()
  var hms = Array[DenseMatrix[Double]]()
  var vvs = Array[DenseVector[Double]]()
  var hvs = Array[DenseVector[Double]]()
  var t   = 0

  def register(ps: Array[DenseMatrix[Double]]) {
    hms = new Array[DenseMatrix[Double]](ps.size)
    vms = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until hms.size) {
      hms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
      vms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
    }
  }

  def register(ps: Array[DenseVector[Double]]) {
    hvs = new Array[DenseVector[Double]](ps.size)
    vvs = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until hvs.size) {
      hvs(i) = DenseVector.zeros[Double](ps(i).size)
      vvs(i) = DenseVector.zeros[Double](ps(i).size)
    }
  }

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    t += 1
    val us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      vms(i) = beta1 * vms(i) + (1d - beta1) * ds(i)
      hms(i) = beta2 * hms(i) + (1d - beta2) * (ds(i) *:* ds(i))
      val vvms = vms(i) / (1d - Math.pow(beta1, t))
      val hhms = hms(i) / (1d - Math.pow(beta2, t))
      us(i) = lr * vvms /:/ (hhms.map(Math.sqrt) + epsilon)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    t += 1
    val us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      vvs(i) = beta1 * vvs(i) + (1d - beta1) * ds(i)
      hvs(i) = beta2 * hvs(i) + (1d - beta2) * (ds(i) *:* ds(i))
      val vvvs = vvs(i) / (1d - Math.pow(beta1, t))
      val hhvs = hvs(i) / (1d - Math.pow(beta2, t))
      us(i) = lr * vvvs /:/ (hhvs.map(Math.sqrt) + epsilon)
    }
    us
  }
}

class RMSProp(val lr: Double = 0.001, val beta: Double = 0.9, val epsilon: Double = 1e-8)
    extends Opt {
  var hms = Array[DenseMatrix[Double]]()
  var hvs = Array[DenseVector[Double]]()

  def register(ps: Array[DenseMatrix[Double]]) {
    hms = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until hms.size) {
      hms(i) = DenseMatrix.zeros[Double](ps(i).rows, ps(i).cols)
    }
  }

  def register(ps: Array[DenseVector[Double]]) {
    hvs = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until hvs.size) {
      hvs(i) = DenseVector.zeros[Double](ps(i).size)
    }
  }

  def update(ps: Array[DenseMatrix[Double]],
             ds: Array[DenseMatrix[Double]]): Array[DenseMatrix[Double]] = {
    val us = new Array[DenseMatrix[Double]](ps.size)
    for (i <- 0 until ps.size) {
      hms(i) = beta * hms(i) + (1d - beta) * (ds(i) *:* ds(i))
      us(i) = lr * ds(i) /:/ (hms(i).map(Math.sqrt) + epsilon)
    }
    us
  }

  def update(ps: Array[DenseVector[Double]],
             ds: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val us = new Array[DenseVector[Double]](ps.size)
    for (i <- 0 until ps.size) {
      hvs(i) = beta * hvs(i) + (1d - beta) * (ds(i) *:* ds(i))
      us(i) = lr * ds(i) /:/ (hvs(i).map(Math.sqrt) + epsilon)
    }
    us
  }
}
