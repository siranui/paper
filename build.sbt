lazy val root = (project in file("."))
  .settings(
    name := "paper"
  )

scalaVersion := "2.12.3"

scalacOptions ++= Seq("-deprecation", "-feature", "-unchecked", "-Xlint")
scalacOptions in (Compile, console) ~= (_.filterNot(_=="-Xlint"))

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze"         % "0.13",
  "org.scalanlp" %% "breeze-natives" % "0.13",
  "org.scalanlp" %% "breeze-viz"     % "0.13",
  "org.scalatest" %% "scalatest" % "3.0.3" % "test",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

resolvers ++= Seq(
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)
