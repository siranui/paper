package pll.actors

import akka.actor.ActorSystem
import breeze.linalg._
import breeze.plot.Figure

object Main extends App {
  import Monitor._
  val system = ActorSystem("system")

  val fig1 = Figure()

  val actor1 = system.actorOf(Monitor.props(fig1),"actor1")

  for(_ <- 0 until 10){
    val w = DenseVector.rand(100) * 100d
    val x = DenseVector.rand(100) * 100d
    val y = DenseVector.rand(100) * 100d
    val z = DenseVector.rand(100) * 100d

    actor1 ! Histgram(Seq(w,x,y,z),2)
    Thread.sleep(300)
    actor1 ! Line(Seq(w,x,y,z),100,2)
    Thread.sleep(300)
    actor1 ! Image(Seq(w,x,y,z),2)

    Thread.sleep(300)
  }

  system.terminate()
  System.exit(0)

}
