package pll

import pll.Im2Col._
import breeze.linalg._
import org.scalatest.FunSuite

class im2colTest extends FunSuite {
  def im2col_check(xw: Int = 4, fw: Int = 2, fset: Int = 2, ch: Int = 2, stride: Int = 2) = {
    val x = convert(DenseVector.range(0, xw * xw * ch), Double)
    // val ff = convert(DenseVector.range(0, fw*fw), Double)
    // val f = Array.ofDim[DenseVector[Double]](fset, ch).map(_.map(_ => ff))

    /* val col = */
    im2col(Array.fill(fset)(x), fw, fw, ch, stride)

    // val fmat = f.map( _.reduce(DenseVector.vertcat(_, _)).toDenseMatrix ).reduce(DenseMatrix.vertcat(_, _))

    // (fmat * col).t.toDenseVector
  }

  def col2im_check(in_w: Int = 4,
                   out_w: Int = 2,
                   fil_w: Int = 2,
                   ch: Int = 2,
                   stride: Int = 2,
                   batch: Int = 2) = {
    val col = reshape(
      DenseVector.range(0, fil_w * fil_w * ch * out_w * out_w * batch).map(_.toDouble),
      out_w * out_w * batch,
      fil_w * fil_w * ch).t

    col2im(col, Shape(batch, ch, in_w, in_w), fil_w, fil_w, stride)
  }

  test("im2col_SCSK_test") {
    val mat = DenseMatrix(
      (0, 1, 3, 4),
      (1, 2, 4, 5),
      (3, 4, 6, 7),
      (4, 5, 7, 8)
    ).map(_.toDouble)
    assert(im2col_check(xw = 3, fw = 2, fset = 1, ch = 1, stride = 1) == mat)
  }

  test("im2col_MCSK_test") {
    val mat = DenseMatrix(
      (0, 1, 3, 4),
      (1, 2, 4, 5),
      (3, 4, 6, 7),
      (4, 5, 7, 8),
      (9, 10, 12, 13),
      (10, 11, 13, 14),
      (12, 13, 15, 16),
      (13, 14, 16, 17)
    ).map(_.toDouble)
    assert(im2col_check(xw = 3, fw = 2, fset = 1, ch = 2, stride = 1) == mat)
  }

  test("im2col_MCMK_test") {
    val mat = DenseMatrix(
      (0, 1, 3, 4, 0, 1, 3, 4),
      (1, 2, 4, 5, 1, 2, 4, 5),
      (3, 4, 6, 7, 3, 4, 6, 7),
      (4, 5, 7, 8, 4, 5, 7, 8),
      (9, 10, 12, 13, 9, 10, 12, 13),
      (10, 11, 13, 14, 10, 11, 13, 14),
      (12, 13, 15, 16, 12, 13, 15, 16),
      (13, 14, 16, 17, 13, 14, 16, 17)
    ).map(_.toDouble)
    assert(im2col_check(xw = 3, fw = 2, fset = 2, ch = 2, stride = 1) == mat)
  }

  test("col2im_SCSK_test") {
    val mat = DenseMatrix(
      (0, 10, 12, 11),
      (21, 62, 66, 43),
      (27, 74, 78, 49),
      (24, 58, 60, 35)
    ).map(_.toDouble)
    assert(col2im_check(in_w = 4, out_w = 3, fil_w = 2, ch = 1, stride = 1, batch = 1)(0)(0) == mat)
  }

}
