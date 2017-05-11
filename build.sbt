name := "deeplearning-tutorial"

version := "2.0"

scalaVersion in Global := "2.11.11"

val deepLearningScalaVersion = "2.0.0-M0"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiable" % deepLearningScalaVersion

libraryDependencies += "org.plotly-scala" %% "plotly-render" % "0.3.2"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.7.2"

libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.7.2"

libraryDependencies += "org.rauschig" % "jarchivelib" % "0.5.0"

libraryDependencies += "com.thoughtworks.each" %% "each" % "3.3.1"

libraryDependencies += "org.slf4j" % "jul-to-slf4j" % "1.7.25"

libraryDependencies += "org.slf4j" % "slf4j-api" % "1.7.25"

libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.2"

libraryDependencies += "ch.qos.logback" % "logback-core" % "1.2.2"

addCompilerPlugin(
  "org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

fork := true
