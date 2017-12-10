package pll.actors

import akka.actor.{Actor, ActorLogging, Props}
import breeze.linalg._
import breeze.plot.Figure

object Monitor {
  def props(fig: Figure) = Props(classOf[Monitor], fig)
//  case class pltInfo(val p: String, xs: Seq[DenseVector[Double]], args: Array[String])
  case class Histgram(xs: Seq[DenseVector[Double]], row: Int, filename: String = "")
  case class Line(xs: Seq[DenseVector[Double]], x_max: Int, row: Int, filename: String = "")
  case class Image_v(xs: Seq[DenseVector[Double]], row: Int, filename: String = "")
  case class Image(xs: Seq[DenseMatrix[Double]], row: Int, filename: String = "")
}

class Monitor(fig: Figure) extends Actor with ActorLogging {
  import Monitor._

  var count = 0
  def receive = {
    case Histgram(xs, row, filename) =>
      pll.graph.Histgram2(fig, xs, row, filename)
      count += 1
      log.info(s"make histgram $count.\n")

    case Line(xs, x_max, row, filename) =>
      pll.graph.Line(fig, xs, x_max, row, filename)
      count += 1
      log.info(s"make plot $count.\n")

    case Image_v(xs, row, filename) =>
      val wid = math.sqrt(xs(0).length).toInt
      val mats = xs.map { v =>
        // 幅が同じならばmapの外で決める。
        // val wid = math.sqrt(v.length).toInt
        reshape(v, wid, wid).t
      }
      pll.graph.Image(fig, mats, row, filename)
      count += 1
      log.info(s"make image $count.\n")

    case Image(xs, row, filename) =>
      pll.graph.Image(fig, xs, row, filename)
      count += 1
      log.info(s"make image $count.\n")

    case _ =>
      log.error("error.\n")
  }

}
