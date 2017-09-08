package pll


import breeze.linalg.DenseVector
import org.scalatest.FunSuite

class PadTest extends FunSuite {

  val ch1: DenseVector[Double] = DenseVector[Double](
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  )

  val ch2: DenseVector[Double] = DenseVector[Double](
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,

    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  )

  // around

  test("ch=1, width=1, around") {
    val p110d = Pad(channel = 1, width = 1)
    val around0 = DenseVector[Double](
      0, 0, 0, 0, 0,
      0, 1, 2, 3, 0,
      0, 4, 5, 6, 0,
      0, 7, 8, 9, 0,
      0, 0, 0, 0, 0
    )
    assert(p110d.forward(ch1) == around0, "around0")
    assert(p110d.backward(p110d.forward(ch1)) == ch1, "around0 back")
    p110d.reset()
  }

  test("ch=2, width=1, around") {
    val p210d = Pad(channel = 2, width = 1)
    val around1 = DenseVector[Double](
      0, 0, 0, 0, 0,
      0, 1, 2, 3, 0,
      0, 4, 5, 6, 0,
      0, 7, 8, 9, 0,
      0, 0, 0, 0, 0,

      0, 0, 0, 0, 0,
      0, 1, 2, 3, 0,
      0, 4, 5, 6, 0,
      0, 7, 8, 9, 0,
      0, 0, 0, 0, 0
    )
    assert(p210d.forward(ch2) == around1, "around1")
    assert(p210d.backward(p210d.forward(ch2)) == ch2, "around1 back")
    p210d.reset()
  }

  test("ch=2, width=2, around") {
    val p220d = Pad(channel = 2, width = 2)
    val around2 = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 2, 3, 0, 0,
      0, 0, 4, 5, 6, 0, 0,
      0, 0, 7, 8, 9, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 2, 3, 0, 0,
      0, 0, 4, 5, 6, 0, 0,
      0, 0, 7, 8, 9, 0, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0
    )
    assert(p220d.forward(ch2) == around2, "around2")
    assert(p220d.backward(p220d.forward(ch2)) == ch2, "around2 back")
    p220d.reset()
  }


  // trans

  test("ch=1, width=1, trans") {
    val t11u = Pad(channel = 1, width = 1, ud = "up")
    val trans1 = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 3, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 4, 0, 5, 0, 6, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 7, 0, 8, 0, 9, 0,
      0, 0, 0, 0, 0, 0, 0
    )
    assert(t11u.forward(ch1) == trans1, "trans1")
    assert(t11u.backward(t11u.forward(ch1)) == ch1, "trans1 back")
    t11u.reset()
  }

  test("ch=2, width=1, trans") {
    val t21u = Pad(channel = 2, width = 1, ud = "up")
    val trans2 = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 3, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 4, 0, 5, 0, 6, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 7, 0, 8, 0, 9, 0,
      0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0,
      0, 1, 0, 2, 0, 3, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 4, 0, 5, 0, 6, 0,
      0, 0, 0, 0, 0, 0, 0,
      0, 7, 0, 8, 0, 9, 0,
      0, 0, 0, 0, 0, 0, 0
    )
    assert(t21u.forward(ch2) == trans2, "trans2")
    assert(t21u.backward(t21u.forward(ch2)) == ch2, "trans2 back")
    t21u.reset()
  }

  test("ch=2, width=2, trans") {
    val t22u = Pad(channel = 2, width = 2, ud = "up")
    val trans22 = DenseVector[Double](
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 4, 0, 0, 5, 0, 0, 6, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    )
    assert(t22u.forward(ch2) == trans22, "trans22")
    assert(t22u.backward(t22u.forward(ch2)) == ch2, "trans22 back")
    t22u.reset()
  }

}
