name := "deeplearning-tutorial"

version := "1.0"

scalaVersion in Global := "2.11.8"

val deepLearningScalaVersion = "1.0.0-RC11"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableany" % deepLearningScalaVersion

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablenothing" % deepLearningScalaVersion

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableseq" % deepLearningScalaVersion

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiabledouble" % deepLearningScalaVersion

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablefloat" % deepLearningScalaVersion

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablehlist" % deepLearningScalaVersion

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablecoproduct" % deepLearningScalaVersion

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableindarray" % deepLearningScalaVersion

libraryDependencies += "org.plotly-scala" %% "plotly-render" % "0.3.1"

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0"

libraryDependencies += "org.nd4j" % "nd4j-cuda-8.0-platform" % "0.8.0"

libraryDependencies += "org.rauschig" % "jarchivelib" % "0.5.0"

addCompilerPlugin(
  "com.thoughtworks.implicit-dependent-type" %% "implicit-dependent-type" % "latest.release")

addCompilerPlugin(
  "org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

fork := true
