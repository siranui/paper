//#full-example
package com.lightbend.akka.sample

import akka.actor.{Actor, ActorLogging, ActorRef, ActorSystem, Props}
import breeze.linalg._
import scala.io.StdIn

//#greeter-companion
//#greeter-messages
object Greeter {
  //#greeter-messages
  def props(message: String, printerActor: ActorRef): Props =
    Props(new Greeter(message, printerActor))
  //#greeter-messages
  final case class Train(args: Array[String])
  final case class Plot(x: DenseVector[Double], counter: Int)
  final case class Plot2(x: List[DenseVector[Double]], counter: Int, i: Int, j: Int)
}
//#greeter-messages
//#greeter-companion

//#greeter-actor
class Greeter(message: String, printerActor: ActorRef) extends Actor {
  import Greeter._
  import Printer._

  def receive = {
    case Train(args) =>
      MNIST.main(args)
    case Plot(x, counter) =>
      printerActor ! PlotHist(x, counter)
    case Plot2(x, counter, i, j) =>
      printerActor ! PlotHist2(x, counter, i, j)
  }
}
//greeter-actor

//#printer-companion
//#printer-messages
object Printer {
  //#printer-messages
  def props: Props = Props[Printer]
  //#printer-messages
  final case class Greeting(greeting: String)
  final case class PlotHist(x: DenseVector[Double], counter: Int)
  final case class PlotHist2(x: List[DenseVector[Double]], counter: Int, i: Int, j: Int)
}
//#printer-messages
//#printer-companion

//#printer-actor
class Printer extends Actor with ActorLogging {
  import Printer._

  def receive = {
    case Greeting(greeting) =>
      log.info(s"Greeting received (from ${sender()}): $greeting")
    case PlotHist(x, counter) =>
    // pll.Bunpu.bunpu(x, counter)
    case PlotHist2(x, counter, i, j) =>
    // pll.Bunpu.bunpu2(x, counter)
    // //Bunpu.bunpu2(vecs.reverse,i,2,(net.layers.size+1)/2)
  }
}
//#printer-actor

//#main-class
object AkkaQuickstart extends App {
  import Greeter._

  // Create the 'helloAkka' actor system
  val system: ActorSystem = ActorSystem("helloAkka")

  try {
    //#create-actors
    // Create the printer actor
    val printer: ActorRef = system.actorOf(Printer.props, "printerActor")

    // Create the 'greeter' actors
    val mnist: ActorRef =
      system.actorOf(Greeter.props("mnist", printer), "mnist")
    //#create-actors

    //#main-send-messages
    mnist ! Train(Array("true"))
    //#main-send-messages

    println(">>> Press ENTER to exit <<<")
    StdIn.readLine()
  } finally {
    system.terminate()
  }
}
//#main-class
//#full-example
