lazy val buildSettings = Seq(
  scalaVersion := "2.12.4",
  logLevel := Level.Info,
  scalacOptions := Seq("-deprecation", "-feature", "-unchecked", "-Xlint"),
  scalacOptions in (Compile, console) ~= (_.filterNot(_ == "-Xlint"))
)

lazy val akkaVersion = "2.5.3"

lazy val scalafmtConfig = file(".scalafmt.conf")

libraryDependencies ++= Seq(
  "org.scalanlp"             %% "breeze"         % "0.13",
  "org.scalanlp"             %% "breeze-natives" % "0.13",
  "org.scalanlp"             %% "breeze-viz"     % "0.13",
  "com.github.fommil.netlib" % "all"             % "1.1.2" pomOnly ()
)

libraryDependencies ++= Seq(
  "com.typesafe.akka"      %% "akka-actor"   % akkaVersion,
  "com.typesafe.akka"      %% "akka-testkit" % akkaVersion,
  "org.scala-lang.modules" %% "scala-swing"  % "2.0.0-M2",
  "org.scalatest"          %% "scalatest"    % "3.0.3" % "test",
)

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

lazy val root = (project in file("."))
  .settings(buildSettings: _*)
  .settings(name := "paper")
