package hyperparameter_tune

import akka.actor._
// import akka.actor.SupervisorStrategy._
// import akka.pattern.ask
import akka.util.Timeout
import breeze.linalg._
import pll.typeAlias._
import scala.concurrent.duration._
// import scala.concurrent.{Await, Future}

case class Conditions(cond: Seq[Any]*) // TODO: パラメータの条件の表し方を考える
case class Condition(cond: Any*)
case class InterimResult(interimResult: Double)
case class Result(cond: Condition, result: Double)

/**
  * Controler class.
  *
  * @param nwb NetworkBuilder
  * @param train_set DATASET of train. Array(Data, Tag)
  * @param test_set DATASET of test. Array(Data, Tag)
  * @param Conditions Conditions that we want to try.
  */
class Controler(nwb: NB, train_set: DATASET, test_set: DATASET, conditions: Conditions)
    extends Actor {
  import context._

  // 子のアクターを生成
  // TODO: Trainerの引数を修正
  val trainer = actorOf(Props(new Trainer(nwb, train_set, test_set)), "trainer")
  context.watch(trainer)

  var num_cond = 0
  var num_fin  = 0

  var resMap = Map[Condition, Double]()

  def receive = {
    case "grid" =>
      // distrs,SDs,update_methods,lrs)
      for {
        distr         <- conditions.cond(0)
        SD            <- conditions.cond(1)
        update_method <- conditions.cond(2)
        lr            <- conditions.cond(3)
      } {
        num_cond += 1
        trainer ! Condition(distr, SD, update_method, lr)
      }
    case Result(c, res) =>
      num_fin += 1
      println(res)
      resMap += c -> res
      if (num_fin == num_cond) {
        context.stop(trainer)
      }
    case Terminated(trainer) =>
      println(resMap.toArray.sortBy(_._2).reverse.take(10).toList)
      println(s"trainer: Terminated.")
      context.stop(self)
  }
}

/**
  * Trainer class.
  * @note this is child actor of "Controler".
  *
  * @param nwb NetworkBuilder
  * @param train_set DATASET of train. Array(Data, Tag)
  * @param test_set DATASET of test. Array(Data, Tag)
  */
class Trainer(nwb: NB, train_set: DATASET, test_set: DATASET) extends Actor {

  def receive = {
    case c: Condition =>
      // TODO: 与えられた条件で学習
      val distr         = c.cond(0).toString
      val SD            = c.cond(1).asInstanceOf[Double]
      val update_method = c.cond(2).toString
      val lr            = c.cond(3).asInstanceOf[Double]

      val net = nwb.apply(distr, SD, update_method, lr)

      val (train_err, train_acc, test_err, test_acc) = mock.train(net, train_set, test_set)

      println(s"$train_err\t${train_acc}\t$test_err\t${test_acc}")

      // TODO: 学習の途中結果を返す
      // sender ! InterimResult(???)

      // TODO: 学習終了時に結果を返す
      sender ! Result(c, test_acc.asInstanceOf[Double])
  }

}

object Main {

  def main(args: Array[String]) {
    mock.args_process(args)

    //----------------------------------------------------------------------------//

    // 3. 探索するハイパーパラメータを列挙
    val distrs         = Seq("Xavier", "He", "Gaussian", "Uniform")
    val SDs            = linspace(0.01, 10, 4).toArray.toSeq
    val update_methods = Seq("SGD", "AdaGrad", "RMSProp", "Adam")
    val lrs            = linspace(0.01, 10, 4).toArray.toSeq

    println("--- data loading ---")
    val Seq((train_data, train_tag), (test_data, test_tag)) = mock.data_load()

    val train_set: DATASET = train_data zip train_tag
    val test_set: DATASET  = test_data zip test_tag

    // var tAccs = Map[(String, Double, String, Double), Double]()

    implicit val timeout = Timeout(1000.milliseconds)

    // actorの生成
    val system = ActorSystem("system")
    val controler = system.actorOf(
      Props(
        new Controler(
          cifar10_net_builder,
          train_set,
          test_set,
          Conditions(distrs, SDs, update_methods, lrs))))

    controler ! "grid"

    // println(system.getState)
    Thread.sleep(100000)

    system.terminate
  }
}
