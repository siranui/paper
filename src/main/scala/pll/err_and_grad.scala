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
    -sum(t *:* breeze.numerics.log(y + 1e-323))
  }

  def calc_cross_entropy_loss(ys: Array[DenseVector[Double]],
                              ts: Array[DenseVector[Double]]): Double = {
    var L = 0d
    for ((y, t) <- ys zip ts) {
      L += -sum(t *:* breeze.numerics.log(y + 1e-323))
    }
    L
  }

  def calc_Lap1_loss(y: DenseVector[Double],
                     t: DenseVector[Double],
                     ch: Int = 1,
                     input_height: Int = 32,
                     input_width: Int = 32): Double = {

    // Pooling
    def Pooling(x: DenseVector[Double]): DenseVector[Double] = {

      val window_height = 2
      val window_width  = 2

      // val xmat = reshape(x, input_height * ch, input_width).t
      val xmat = reshape(x, input_width, input_height * ch).t

      val mx = (for {
        i <- 0 until input_height * ch by window_height
        j <- 0 until input_width by window_width
      } yield {
        max(xmat(i until i + window_height, j until j + window_width))
      }).toArray

      DenseVector(mx)
    }

    // up-sampling
    def UpSampling(x: DenseVector[Double]): DenseVector[Double] = {
      val sz                             = 2
      val xs: Array[DenseVector[Double]] = utils.divideIntoN(x, ch)
      val upsample_mats: Array[DenseVector[Double]] = xs.map { m =>
        val reshaped_m = reshape(m, input_height / 2, input_width / 2).t
        val upsample_mat = DenseMatrix.tabulate(input_height / 2 * sz, input_width / 2 * sz) {
          case (i, j) => reshaped_m(i / sz, j / sz)
        }
        upsample_mat.t.toDenseVector
      }
      // println(s"${upsample_mats.toList}".debug)

      upsample_mats.reduce(DenseVector.vertcat(_, _))
    }

    sum(breeze.numerics.abs(UpSampling(Pooling(y)) - UpSampling(Pooling(t)))) / 4d
  }

  def calc_Lap1_loss(ys: Array[DenseVector[Double]], ts: Array[DenseVector[Double]]): Double = {
    var L = 0d
    for ((y, t) <- ys zip ts) {
      L += calc_Lap1_loss(y, t)
    }
    L
  }

}

object grad {
  def calc_L2_grad(y: DenseVector[Double], t: DenseVector[Double]): DenseVector[Double] = {
    y -:- t
  }

  def calc_L2_grad(ys: Array[DenseVector[Double]],
                   ts: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val grads = for ((y, t) <- ys zip ts) yield {
      y -:- t
    }
    grads
  }

  def calc_cross_entropy_grad(y: DenseVector[Double],
                              t: DenseVector[Double]): DenseVector[Double] = {
    y -:- t
  }

  def calc_cross_entropy_grad(ys: Array[DenseVector[Double]],
                              ts: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val grads = for ((y, t) <- ys zip ts) yield {
      y -:- t
    }
    grads
  }

  def calc_Lap1_grad(y: DenseVector[Double],
                     t: DenseVector[Double],
                     ch: Int = 1,
                     input_height: Int = 32,
                     input_width: Int = 32): DenseVector[Double] = {

    // Pooling
    def Pooling(x: DenseVector[Double]): DenseVector[Double] = {

      val window_height = 2
      val window_width  = 2

      // val xmat = reshape(x, input_height * ch, input_width).t
      val xmat = reshape(x, input_width, input_height * ch).t

      val mx = (for {
        i <- 0 until input_height * ch by window_height
        j <- 0 until input_width by window_width
      } yield {
        max(xmat(i until i + window_height, j until j + window_width))
      }).toArray

      DenseVector(mx)
    }

    // up-sampling
    def UpSampling(x: DenseVector[Double]): DenseVector[Double] = {
      val sz                             = 2
      val xs: Array[DenseVector[Double]] = utils.divideIntoN(x, ch)
      val upsample_mats: Array[DenseVector[Double]] = xs.map { m =>
        val reshaped_m = reshape(m, input_height / 2, input_width / 2).t
        val upsample_mat = DenseMatrix.tabulate(input_height / 2 * sz, input_width / 2 * sz) {
          case (i, j) => reshaped_m(i / sz, j / sz)
        }
        upsample_mat.t.toDenseVector
      }
      // println(s"${upsample_mats.toList}".debug)

      upsample_mats.reduce(DenseVector.vertcat(_, _))
    }

    UpSampling(Pooling(y)) - UpSampling(Pooling(t))
  }

  def calc_Lap1_grad(ys: Array[DenseVector[Double]],
                     ts: Array[DenseVector[Double]]): Array[DenseVector[Double]] = {
    val grads = for ((y, t) <- ys zip ts) yield {
      calc_Lap1_grad(y, t)
    }
    grads
  }

}
