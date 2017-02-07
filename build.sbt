name := "deeplearning-tutorial"

version := "1.0"

scalaVersion in Global := "2.11.8"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableany" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablenothing" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableseq" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiabledouble" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablefloat" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablehlist" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiablecoproduct" % "latest.release"

libraryDependencies += "com.thoughtworks.deeplearning" %% "differentiableindarray" % "latest.release"

addCompilerPlugin("com.thoughtworks.implicit-dependent-type" %% "implicit-dependent-type" % "latest.release")

addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

fork := true
    