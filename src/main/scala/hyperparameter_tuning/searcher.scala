package hyperparameter_tune

import akka.actor._
// import akka.actor.SupervisorStrategy._
// import akka.pattern.ask
import akka.util.Timeout
import breeze.linalg._
import pll.typeAlias._
import scala.concurrent.duration._
import scala.concurrent.Await

case class Conditions(cond: Seq[Any]*)
case class Condition(cond: Any*)
case class InterimResult(interimResult: Double)
case class Result(cond: Condition, result: Double)

object Controler {
  def props(nwb: NB, train_set: DATASET, test_set: DATASET, conditions: Conditions)(
      implicit nrOfThread: Int = 4) =
    Props(new Controler(nwb, train_set, test_set, conditions)(nrOfThread))
}

/**
  * Controler class.
  *
  * @param nwb NetworkBuilder
  * @param train_set DATASET of train. Array(Data, Tag)
  * @param test_set DATASET of test. Array(Data, Tag)
  * @param Conditions Conditions that we want to try.
  *
  * @param nrOfThread number of thread.
  */
class Controler(nwb: NB, train_set: DATASET, test_set: DATASET, conditions: Conditions)(
    implicit nrOfThread: Int = 4)
    extends Actor
    with ActorLogging {
  import context._

  def createTrainer(name: String) = {
    context.actorOf(Trainer.props(nwb, train_set, test_set), name)
    // context.actorOf(Props(new Trainer(nwb, train_set, test_set)), name)
  }

  // Generate Chile Actor.
  val trainers = (0 until nrOfThread).map(i => createTrainer("trainer-" + i))
  trainers.map(trainer => context.watch(trainer))

  var terminated = 0

  var resMap = Map[Condition, Double]()

  def receive = {
    case "grid" =>
      // distrs,SDs,update_methods,lrs)
      var i = 0
      for {
        distr         <- conditions.cond(0)
        SD            <- conditions.cond(1)
        update_method <- conditions.cond(2)
        lr            <- conditions.cond(3)
      } {
        trainers(i % nrOfThread) ! Condition(distr, SD, update_method, lr)

        i += 1
        if (i > 10000) i = 0
      }
    case Result(c, res) =>
      // println(res)
      resMap += c -> res
      sender() ! PoisonPill // stop child actor
    case Terminated(actorRef) =>
      terminated += 1

      log.info(s"nrOfTerminated: $terminated")

      if (terminated == nrOfThread) { // trainerが全て終了した時の処理
        println("\n\n\n")
        resMap.toArray
          .sortBy(_._2)
          .reverse
          .take(10)
          .toList
          .foreach(p => println(s"conditon:${p._1.toString}\ttest acc:${p._2}"))
        println("\n\n\n")
        log.info(s"trainer: Terminated.")
        println("\n\n\n")
        // context.stop(self)
        context.system.terminate()
      }
  }
}

object Trainer {
  def props(nwb: NB, train_set: DATASET, test_set: DATASET) =
    Props(new Trainer(nwb, train_set, test_set))
}

/**
  * Trainer class.
  * @note this is child actor of "Controler".
  *
  * @param nwb NetworkBuilder
  * @param train_set DATASET of train. Array(Data, Tag)
  * @param test_set DATASET of test. Array(Data, Tag)
  */
class Trainer(nwb: NB, train_set: DATASET, test_set: DATASET) extends Actor with ActorLogging {

  def receive = {
    case c: Condition =>
      // 与えられた条件で学習
      val distr         = c.cond(0).toString
      val SD            = c.cond(1).asInstanceOf[Double]
      val update_method = c.cond(2).toString
      val lr            = c.cond(3).asInstanceOf[Double]

      val net = nwb.apply(distr, SD, update_method, lr)

      val (train_err, train_acc, test_err, test_acc) = mock.train(net, train_set, test_set)

      // println(s"$distr, $SD, $update_method, $lr")
      // println(s"$train_err\t${train_acc}\t$test_err\t${test_acc}")

      // TODO: 学習の途中結果を返す
      // sender ! InterimResult(???)

      // Return result when training finished.
      sender ! Result(c, test_acc.asInstanceOf[Double])
  }

}

object Main {

  def main(args: Array[String]) {

    mock.args_process(args)

    // 探索するハイパーパラメータを列挙
    val distrs         = Seq("Xavier", "He", "Gaussian", "Uniform")
    val SDs            = linspace(0.01, 10, 4).toArray.toSeq
    val update_methods = Seq("SGD", "AdaGrad", "RMSProp", "Adam")
    val lrs            = linspace(0.01, 10, 4).toArray.toSeq

    println("--- data loading ---")
    val Seq((train_data, train_tag), (test_data, test_tag)) = mock.data_load()

    val train_set: DATASET = train_data zip train_tag
    val test_set: DATASET  = test_data zip test_tag

    implicit val timeout = Timeout(1000.milliseconds)

    // Generate Actor System
    val system = ActorSystem("system")
    val controler = system.actorOf(
      Props(
        new Controler(
          cifar10_net_builder,
          train_set,
          test_set,
          Conditions(distrs, SDs, update_methods, lrs))(
          nrOfThread = 4
        )
      ))

    controler ! "grid"

    // system.terminate が呼ばれるまで待機する
    Await.ready(system.whenTerminated, Duration.Inf)
    println("\n\n\nComplete.\n\n\n")

  }
}
