package pll

import breeze.linalg._
import org.scalatest.FunSuite

class PoolingTest extends FunSuite {
  case class Shape(N: Int, C: Int, H: Int, W: Int)

  val ch1: DenseVector[Double] = DenseVector[Double](
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36
  )

  val ch2: DenseVector[Double] = DenseVector[Double](
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
  )

  val rec: DenseVector[Double] = DenseVector[Double](
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
  )

  val rec2: DenseVector[Double] = DenseVector[Double](
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3,
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
  )

  // max-pooling
  test("max-pooling, channel=1, width=2") {
    val pool = Pooling(2, 2)(1, 6, 6)
    val pooled = DenseVector[Double](
      8, 10, 12, 20, 22, 24, 32, 34, 36
    )
    assert(pool.forward(ch1) == pooled, "max-pooling(ch=1, width=2): forward error.")

    val pooled_back = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0, 8, 0, 10, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 22, 0, 24, 0, 0, 0, 0, 0, 0,
      0, 32, 0, 34, 0, 36
    )
    assert(pool.backward(pooled) == pooled_back, "max-pooling(ch=1, width=2): backward error.")
  }

  test("max-pooling, channel=1, width=3") {
    val pool = Pooling(3, 3)(1, 6, 6)
    val pooled = DenseVector[Double](
      15,
      18,
      33,
      36
    )
    assert(pool.forward(ch1) == pooled, "max-pooling(ch=1, width=3): forward error.")

    val pooled_back = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 33, 0, 0, 36
    )
    assert(pool.backward(pooled) == pooled_back, "max-pooling(ch=1, width=3): backward error.")
  }

  test("max-pooling, channel=2, width=3") {
    val pool = Pooling(3, 3)(2, 6, 6)
    val pooled = DenseVector[Double](
      15, 18, 33, 36, 15, 18, 33, 36
    )
    assert(pool.forward(ch2) == pooled, "max-pooling(ch=2, width=3): forward error.")

    val pooled_back = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 33, 0, 0, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 33, 0, 0, 36
    )
    assert(pool.backward(pooled) == pooled_back, "max-pooling(ch=2, width=3): backward error.")
  }

  test("max-pooling(rectangle), channel=1, width=2") {
    val pool = Pooling(2, 2)(1, 4, 6)
    val pooled = DenseVector[Double](
      8, 10, 12, 20, 22, 24,
    )
    assert(pool.forward(rec) == pooled, "max-pooling(ch=1, width=2): forward error.")

    val pooled_back = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0, 8, 0, 10, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 22, 0, 24,
    )
    assert(pool.backward(pooled) == pooled_back, "max-pooling(ch=1, width=2): backward error.")
  }

  test("max-pooling(rectangle), channel=2, width=2") {
    val pool = Pooling(2, 2)(2, 4, 6)
    val pooled = DenseVector[Double](
      8, 10, 12, 20, 22, 24, 8, 10, 12, 20, 22, 24,
    )
    assert(pool.forward(rec2) == pooled, "max-pooling(ch=1, width=2): forward error.")

    val pooled_back = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0, 8, 0, 10, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 22, 0, 24, 0, 0, 0, 0, 0, 0,
      0, 8, 0, 10, 0, 12, 0, 0, 0, 0, 0, 0, 0, 20, 0, 22, 0, 24,
    )
    assert(pool.backward(pooled) == pooled_back, "max-pooling(ch=1, width=2): backward error.")
  }
}
