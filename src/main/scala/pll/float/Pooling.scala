package pll.float

import breeze.linalg._
import typeAlias._
// import breeze.numerics._

case class Pooling(window_height: Int, window_width: Int)(ch: Int,
                                                          input_height: Int,
                                                          input_width: Int)
    extends Layer {

  var input = List[DenseVector[T]]()

  def forward(x: DV): DV = {
    this.input = x :: this.input
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

  def backward(d: DV): DV = {
    val in = this.input.head
    this.input = this.input.tail

    val in_mat = reshape(in, input_width, input_height * ch).t
    val retval = DenseMatrix.zeros[T](in_mat.rows, in_mat.cols)
    for {
      i <- 0 until input_height * ch by window_height
      j <- 0 until input_width by window_width
    } {
      val mx = argmax(in_mat(i until i + window_height, j until j + window_width))
      retval(mx._1 + i, mx._2 + j) += d(
        (i / window_height) * (input_width / window_width) + j / window_width)
    }
    retval.t.toDenseVector
  }

  def update(): Unit = {}

  def reset(): Unit = {}

  def save(filename: String): Unit = {}

  def load(filename: String): Unit = {}

  override def load(data: List[String]): List[String] = { data }
}
