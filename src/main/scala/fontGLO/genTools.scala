package fontGLO

import breeze.linalg._
import pll.typeAlias._

object genTools {
  def Morphing(A: DV, B: DV)(implicit nrOfGenImg: Int = 100): Seq[DV] = {
    for {
      i <- 0 until nrOfGenImg
    } yield {
      A * (1d - i / nrOfGenImg.toDouble) + B * (i / nrOfGenImg.toDouble)
    }
  }

}
