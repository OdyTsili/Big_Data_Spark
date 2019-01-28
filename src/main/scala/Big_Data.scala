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
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.ml.param.{ParamMap,Param}
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.{Dataset,DataFrame}






class CosineTrasfromer(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("str"))
  final val inputCols = new Param[Array[String]](this, "inputCols", "The input column")
  final val outputCol = new Param[String](this, "outputCol", "The output column")
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  override def transform(dataset: Dataset[_]): DataFrame = {

    val sparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("Transdormer")
      .getOrCreate()
    sparkSession.sparkContext.setLogLevel("ERROR")

    val rows2 = new VectorAssembler().setInputCols($(inputCols)).setOutputCol("vecWords")
      .transform(dataset).select("vecWords").rdd
    val items_mllib_vector2 = rows2.map(_.getAs[org.apache.spark.ml.linalg.Vector](0))
      .map(org.apache.spark.mllib.linalg.Vectors.fromML)
    val mat2 = new RowMatrix(items_mllib_vector2)
    val simsPerfect2 = mat2.columnSimilarities()
    val transrormedRDD2 = simsPerfect2.entries.map { case MatrixEntry(row: Long, col: Long, sim: Double) => (row, col, sim) }
    val dfTr = sparkSession.createDataFrame(transrormedRDD2).toDF("ID_111", "ID_222", "cosSim")
    dfTr.createOrReplaceTempView("dfTr")

    val ds = dfTr.join(dataset,(dfTr("ID_111"))===dataset("ID_1")).select("label","ID_111",
      "N1remWords","ID_222","cosSim")
    val dss  = ds
      .withColumnRenamed("label","label1")
      .withColumnRenamed("ID_111","ID_11")
      .withColumnRenamed("N1remWords","N11remWords")
      .withColumnRenamed("ID_222","ID_22")
      .withColumnRenamed("cosSim","cS")
    val ds1 =   dss.join(dataset,dss("ID_22")===dataset("ID_2")).select("label1","ID_11","N11remWords","ID_22",
    "N2remWords","cS")
    sparkSession.stop()
    ds1.distinct().select(ds1("label1").as("label"),ds1("N11remWords").as("N1remWords"),
      ds1("ID_22").as("ID_2"),ds1("N2remWords"),ds1("cS").as("cosineSimilarity"))
  }
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
  override def transformSchema(schema: StructType): StructType = {
    val inputColNames = $(inputCols)
    val outputColNames = $(outputCol)
    schema.add(StructField($(outputCol), DoubleType, false))
  }
}




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
    //val fractions = Map(1.0 -> .001, 0.0 -> .0017)
    val fTrain = filledTrainDF.stat.sampleBy("label", fractions, 36L)
    fTrain.groupBy("label").count().show()

    //set up the first pipeline
    val sqlDF = new SQLTransformer().setStatement("SELECT DISTINCT label,ID_1,ID_2,CONCAT_WS(',',ID_1, pYearN1," +
      "titleN1,authorsN1) AS N1,CONCAT_WS(',',ID_2,pYearN2,titleN2,authorsN2) AS N2 FROM __THIS__")
    val tokDF1 = new Tokenizer().setInputCol("N1").setOutputCol("N1tokWords")
    val tokDF2 = new Tokenizer().setInputCol("N2").setOutputCol("N2tokWords")
    val remDF1 = new StopWordsRemover().setInputCol("N1tokWords").setOutputCol("N1remWords")
    val remDF2 = new StopWordsRemover().setInputCol("N2tokWords").setOutputCol("N2remWords")
    val tfDF1 = new HashingTF().setInputCol("N1remWords").setOutputCol("N1tfWords")
    val tfDF2 = new HashingTF().setInputCol("N2remWords").setOutputCol("N2tfWords")
    val idfDF1 = new IDF().setInputCol("N1tfWords").setOutputCol("N1idfWords")
    val idfDF2 = new IDF().setInputCol("N2tfWords").setOutputCol("N2idfWords")
    val cosDF = new CosineTrasfromer().setInputCols(Array("N1idfWords","N2idfWords")).setOutputCol("cosineSimilarity")

    val w2vDF1 = new Word2Vec().setInputCol("N1remWords").setOutputCol("N1w2vWords").setSeed(36L).setMinCount(2)
    val w2vDF2 = new Word2Vec().setInputCol("N2remWords").setOutputCol("N2w2vWords").setSeed(36L).setMinCount(2)

    val Asmblr = new VectorAssembler().setInputCols(Array("ID_1","ID_2","cosineSimilarity")).setOutputCol("features")
    val dTree = new DecisionTreeClassifier()
      .setLabelCol("label").setFeaturesCol("features")

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(sqlDF, tokDF1,tokDF2,remDF1,remDF2,tfDF1,tfDF2,idfDF1,idfDF2,cosDF,Asmblr,dTree))


   // Search through decision tree's maxDepth parameter for best model
    val paramGrid = new ParamGridBuilder()
      .addGrid(dTree.maxDepth, Array(4, 5, 6))
      //.addGrid(dTree.maxBins, Array(35, 49, 52, 55))
      //.addGrid(tfDF1.numFeatures, Array(20,30,45,55,65))
      //.addGrid(tfDF2.numFeatures, Array(20,30,45,55,65))
      //.addGrid(dTree.impurity, Array("entropy", "gini"))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")

    // Set up 3-fold cross validation
    val crossval = new CrossValidator().setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid).setNumFolds(3)

    val cvModel = crossval.fit(fTrain)

    val bestModel = cvModel.bestModel
    println("The Best Model and Parameters:\n--------------------")
    println(bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(11))
    bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel]
      .stages(11)
      .extractParamMap

    val treeModel = bestModel.asInstanceOf[org.apache.spark.ml.PipelineModel].stages(11).asInstanceOf[DecisionTreeClassificationModel]
    println("Learned classification tree model:\n" + treeModel.toDebugString)

    val predictions = cvModel.transform(info_truthDF)
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