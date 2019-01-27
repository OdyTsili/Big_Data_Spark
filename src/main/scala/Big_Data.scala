import org.apache.spark._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import ml.feature.SQLTransformer
import org.apache.spark.mllib.linalg.distributed.{MatrixEntry, RowMatrix}



object Big_Data {
  val trainSchema = StructType(Array(
    StructField("ID_1", DoubleType, true),
    StructField("ID_2", DoubleType, true),
    StructField("label", DoubleType, true)
  ))
  val infoSchema = StructType(Array(
    StructField("ID", DoubleType, true),
    StructField("pub_year", DoubleType, true),
    StructField("title", StringType, true),
    StructField("authors", StringType, true),
    StructField("journal", StringType, true),
    StructField("abstract", StringType, true)
  ))
  val truthSchema = StructType(Array(
    StructField("ID_1", DoubleType, true),
    StructField("ID_2", DoubleType, true)
  ))

  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.executor.memory", "70g")
      .config("spark.driver.memory", "50g")
      .config("spark.memory.offHeap.enabled", true)
      .config("spark.memory.offHeap.size", "16g")
      .appName("Big_Data")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    val trainDF = spark.read
      .option("delimiter", " ")
      .schema(trainSchema)
      .csv("src/main/resources/training_set.txt").persist()
    trainDF.createOrReplaceTempView("train_data")


    val infoDF = spark.read
      .option("delimiter", ",")
      .schema(infoSchema)
      .csv("src/main/resources/node_information.csv").persist()
    infoDF.createOrReplaceTempView("info_data")


    val trDF = spark.read
      .option("delimiter", "\t")
      .schema(truthSchema)
      .csv("src/main/resources/Cit-HepTh.txt")
    val truthDF = trDF.withColumn("label", typedLit(1.0))
    truthDF.createOrReplaceTempView("truth_data")

    //join links with their information for training data
    val info_truthDF = spark.sql("SELECT t.label,t.ID_1,  t.pub_year AS pYearN1,t.title AS titleN1," +
      "t.authors AS authorsN1, t.ID_2,c.pub_year AS pYearN2, c.title AS titleN2, c.authors AS authorsN2" +
      " FROM(SELECT * FROM truth_data AS a,info_data AS b WHERE  a.ID_1=b.ID) as t, info_data AS c WHERE t.ID_2=c.ID").cache()
    info_truthDF.createOrReplaceTempView("info_truthDF")
    println("------------TruthDF------------")
    info_truthDF.show()


    //join links with their information for validation data
    val info_trainDF = spark.sql("SELECT t.label,t.ID_1,  t.pub_year AS pYearN1,t.title AS titleN1," +
      "t.authors AS authorsN1, t.ID_2,c.pub_year AS pYearN2, c.title AS titleN2, c.authors AS authorsN2" +
      " FROM(SELECT * FROM train_data AS a,info_data AS b WHERE  a.ID_1=b.ID) as t, info_data AS c WHERE t.ID_2=c.ID").cache()
    info_trainDF.createOrReplaceTempView("info_trainDF")

    //fill with '0.0' whichever labels have NaN values
    val filledTrainDF = info_trainDF.na.fill(0.0, Seq("label"))
    println("------------TrainDF------------")
    filledTrainDF.show()


    filledTrainDF.groupBy("label").count().show()
    //downsampling the 1.0 values OR prouning the data
    val fractions = Map(1.0 -> .836, 0.0 -> 1.0)
    val frTrain = filledTrainDF.stat.sampleBy("label", fractions, 36L)
    frTrain.groupBy("label").count().show()
    frTrain.createTempView("trainData")

    //concatenate columns{"year","title","author"} for each node
    val concTrain = spark.sql("SELECT DISTINCT label,ID_1,ID_2,CONCAT_WS(',',ID_1, pYearN1,titleN1,authorsN1) AS N1," +
      "CONCAT_WS(',',ID_2,pYearN2,titleN2,authorsN2) AS N2 FROM trainData")
    println("------------concatenated Train Data------------")
    concTrain.show()

    //concatenate columns{"year","title","author"} for each node
    val concTruth = spark.sql("SELECT DISTINCT label,ID_1,ID_2,CONCAT_WS(',',ID_1, pYearN1,titleN1,authorsN1) AS N1," +
      "CONCAT_WS(',',ID_2,pYearN2,titleN2,authorsN2) AS N2 FROM info_truthDF")
    println("------------concatenated Truth Data------------")
    concTruth.show()

    //set up the first pipeline
    val tokDF1 = new Tokenizer().setInputCol("N1").setOutputCol("N1tokWords")
    val tokDF2 = new Tokenizer().setInputCol("N2").setOutputCol("N2tokWords")
    val remDF1 = new StopWordsRemover().setInputCol("N1tokWords").setOutputCol("N1remWords")
    val remDF2 = new StopWordsRemover().setInputCol("N2tokWords").setOutputCol("N2remWords")
    val tfDF1 = new HashingTF().setInputCol("N1remWords").setOutputCol("N1tfWords")
    val tfDF2 = new HashingTF().setInputCol("N2remWords").setOutputCol("N2tfWords")
    val idfDF1 = new IDF().setInputCol("N1tfWords").setOutputCol("N1idfWords")
    val idfDF2 = new IDF().setInputCol("N2tfWords").setOutputCol("N2idfWords")

    val myPipeline = new Pipeline()
      .setStages(Array(tokDF1, tokDF2, remDF1, remDF2, tfDF1, tfDF2, idfDF1, idfDF2))
    //transorm the firtst pipeline for training data
    val dftrain = myPipeline.fit(concTrain)
    val dfftrain = dftrain.transform(concTrain).drop("N1").drop("N2").drop("N1tokWords")
      .drop("N2tokWords").drop("N1tfWords").drop("N2tfWords")
    dfftrain.createOrReplaceTempView("dfftrain")
    println("------------IDF Represenation of Train Data-----------")
    dfftrain.show()

    //transorm the firtst pipeline for validation data
    val dftruth = myPipeline.fit(concTruth)
    val dfftruth = dftruth.transform(concTruth).drop("N1").drop("N2").drop("N1tokWords")
      .drop("N2tokWords").drop("N1tfWords").drop("N2tfWords")
    dfftruth.createOrReplaceTempView("dfftruth")
    println("------------IDF Represenation of Ground Truth Data-----------")
    dfftruth.show()

    //compute cosine Similarity (couldn't provide it unfortunately as a custom transformer)
    val inputCols = Array("N1idfWords", "N2idfWords")
    val rows1 = new VectorAssembler().setInputCols(inputCols).setOutputCol("vecWords")
      .transform(dfftrain).select("vecWords").rdd
    val items_mllib_vector1 = rows1.map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
      .map(org.apache.spark.mllib.linalg.Vectors.fromML)
    val mat1 = new RowMatrix(items_mllib_vector1)
    val simsPerfect1 = mat1.columnSimilarities()
    val transrormedRDD1 = simsPerfect1.entries.map { case MatrixEntry(row: Long, col: Long, sim: Double) => (row, col, sim) }
    val dfTra = spark.createDataFrame(transrormedRDD1).toDF("ID_1", "ID_2", "cosineSimilarity")
    dfTra.createOrReplaceTempView("dfTra")

    //compute cosine Similarity (couldn't provide it unfortunately as a custom transformer)
    val rows2 = new VectorAssembler().setInputCols(inputCols).setOutputCol("vecWords")
      .transform(dfftruth).select("vecWords").rdd
    val items_mllib_vector2 = rows2.map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
      .map(org.apache.spark.mllib.linalg.Vectors.fromML)
    val mat2 = new RowMatrix(items_mllib_vector2)
    val simsPerfect2 = mat2.columnSimilarities()
    val transrormedRDD2 = simsPerfect2.entries.map { case MatrixEntry(row: Long, col: Long, sim: Double) => (row, col, sim) }
    val dfTru = spark.createDataFrame(transrormedRDD2).toDF("ID_1", "ID_2", "cosineSimilarity")
    dfTru.createOrReplaceTempView("dfTru")

    //set up the second pipeline, which will be the only one to be provided to CV
    val featureCols = Array("N1w2vWords", "N2w2vWords", "cosineSimilarity")
    val sqlDF1 = new SQLTransformer().setStatement(" SELECT DISTINCT s.label, s.ID_1, s.N1remWords,d.ID_2, s.N2remWords," +
      "s.cosineSimilarity FROM(SELECT a.label, a.ID_1,a.ID_2, a.N1remWords, a.N2remWords," +
      "b.ID_1 AS ID_11,b.ID_2 AS ID_22,b.cosineSimilarity FROM dfftrain a, dfTra b WHERE b.ID_1=a.ID_1 )AS s, dfTra as d WHERE " +
      "d.ID_2=s.ID_2 ")
    val sqlTrainDF = sqlDF1.transform(dfftrain)
    println("------------Cosine Similarity Between N1 - N2 on Train Data-----------")
    sqlTrainDF.show()

    val sqlDF2 = new SQLTransformer().setStatement(" SELECT DISTINCT s.label, s.ID_1, s.N1remWords,d.ID_2, s.N2remWords," +
      "s.cosineSimilarity FROM(SELECT a.label, a.ID_1,a.ID_2, a.N1remWords, a.N2remWords," +
      "b.ID_1 AS ID_11,b.ID_2 AS ID_22,b.cosineSimilarity FROM dfftruth a, dfTru b WHERE b.ID_1=a.ID_1 )AS s, dfTru as d WHERE " +
      "d.ID_2=s.ID_2 ")
    val sqlTruthDF = sqlDF2.transform(dfftruth)
    println("------------Cosine Similarity Between N1 - N2 on Groung Truth Data-----------")
    sqlTruthDF.show()


    val w2vDF1 = new Word2Vec().setInputCol("N1remWords").setOutputCol("N1w2vWords").setSeed(36L).setMinCount(2)
    val w2vDF2 = new Word2Vec().setInputCol("N2remWords").setOutputCol("N2w2vWords").setSeed(36L).setMinCount(2)

    val Asmblr = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    val dTree = new DecisionTreeClassifier()
      .setLabelCol("label").setFeaturesCol("features")

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(w2vDF1, w2vDF2, Asmblr, dTree))


    // Search through decision tree's maxDepth parameter for best model
    val paramGrid = new ParamGridBuilder()
      .addGrid(dTree.maxDepth, Array(4, 5, 6, 7))
      //.addGrid(dTree.maxBins, Array(35, 49, 52, 55))
      //.addGrid(hashTF.numFeatures, Array(20,30,45,55,65))
      //.addGrid(dTree.impurity, Array("entropy", "gini"))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    // Set up 3-fold cross validation
    val crossval = new CrossValidator().setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid).setNumFolds(3)

    val cvModel = crossval.fit(sqlTrainDF)

    val bestModel = cvModel.bestModel
    println("The Best Model and Parameters:\n--------------------")
    println(bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(3))
    bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(3)
      .extractParamMap

    val treeModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(3).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)

    val predictions = cvModel.transform(sqlTruthDF)
    val accuracy = evaluator.evaluate(predictions)
    evaluator.explainParams()

    val predictionAndLabels = predictions.select("prediction", "label").rdd.map(x =>
      (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double]))
    val metrics = new BinaryClassificationMetrics(predictionAndLabels)
    println("area under the precision-recall curve: " + metrics.areaUnderPR)
    println("area under the receiver operating characteristic (ROC) curve : " + metrics.areaUnderROC)

    println(metrics.fMeasureByThreshold())

    val result = predictions.select("label", "prediction", "probability")
    result.show

    val lp = predictions.select("label", "prediction")
    val counttotal = predictions.count()
    val correct = lp.filter($"label" === $"prediction").count()
    val wrong = lp.filter(not($"label" === $"prediction")).count()
    val ratioWrong = wrong.toDouble / counttotal.toDouble
    val ratioCorrect = correct.toDouble / counttotal.toDouble
    val truep = lp.filter($"prediction" === 1.0).filter($"label" === $"prediction").count() / counttotal.toDouble
    val truen = lp.filter($"prediction" === 0.0).filter($"label" === $"prediction").count() / counttotal.toDouble
    val falsep = lp.filter($"prediction" === 1.0).filter(not($"label" === $"prediction")).count() / counttotal.toDouble
    val falsen = lp.filter($"prediction" === 0.0).filter(not($"label" === $"prediction")).count() / counttotal.toDouble

    val precisionScore = truep / (truep + falsep)
    val recallScore = truep / (truep + falsen)


    println("counttotal : " + counttotal)
    println("correct : " + correct)
    println("wrong: " + wrong)
    println("ratio wrong: " + ratioWrong)
    println("ratio correct: " + ratioCorrect)
    println("ratio true positive : " + truep)
    println("ratio false positive : " + falsep)
    println("ratio true negative : " + truen)
    println("ratio false negative : " + falsen)
    println("precision:" + precisionScore)
    println("recall:" + recallScore)
    println("f1:" + 2 * ((precisionScore * recallScore) / (precisionScore + recallScore)))

    println("wrong: " + wrong)

    val equalp = predictions.selectExpr(
      "CAST(round(prediction)AS float) as prediction", "label",
      """CASE CAST(round(prediction)AS float) = label WHEN true then 1 ELSE 0 END as equal"""
    )
    equalp.show


  }
}