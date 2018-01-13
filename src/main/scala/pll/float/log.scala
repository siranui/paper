package pll.float

object log {
  import java.util.Date
  def now = "%tF %<tT.%<tL" format new Date

  var logLevel = 3

  // color
  val black   = "\u001b[30m"
  val red     = "\u001b[31m"
  val green   = "\u001b[32m"
  val yellow  = "\u001b[33m"
  val blue    = "\u001b[34m"
  val magenta = "\u001b[35m"
  val cyan    = "\u001b[36m"
  val white   = "\u001b[37m"

  val reset = "\u001b[0m"

  // message
  def error(msg: String) = if (logLevel >= 0) println(s"[${red}error${reset}] $msg [$now]")
  def warn(msg: String)  = if (logLevel >= 1) println(s"[${yellow}warn${reset}] $msg [$now]")
  def info(msg: String)  = if (logLevel >= 2) println(s"[${magenta}info${reset}] $msg [$now]")
  def debug(msg: String) = if (logLevel >= 3) println(s"[${cyan}debug${reset}] $msg")
}
