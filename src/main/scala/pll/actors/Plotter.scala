package pll.actors


import akka.actor.{Actor, ActorLogging, Props}
import breeze.linalg._
import breeze.plot.Figure

object Plotter {
  def props(fig: Figure) = Props(classOf[Plottor], fig)
  case class pltInfo(val p: String, xs: Seq[DenseVector[Double]], args: Array[String])

}

class Plottor(fig: Figure) extends Actor with ActorLogging {
  import Plotter._

  var count = 0
  def receive = {
    case pltInfo("hist",xs,args) =>
      pll.graph.Histgram2(fig, xs, args(0).toInt, args(1))
      count += 1
      log.info(s"make histgram $count.\n")
    case pltInfo("plot",xs,args) =>
      pll.graph.Plot(fig, xs, args(0).toInt, args(1).toInt, args(2))
      count += 1
      log.info(s"make plot $count.\n")
    case pltInfo("image",xs,args) =>
      val wid = math.sqrt(xs(0).length).toInt
      val mats = xs.map{v =>
        // 幅が同じならばmapの外で決める。
        // val wid = math.sqrt(v.length).toInt
        reshape(v,wid,wid).t
      }
      pll.graph.Image(fig, mats, args(0).toInt, args(1))
      count += 1
      log.info(s"make image $count.\n")
    case _ =>
      log.error("error.\n")
  }

}

