# Auto-generated Spark Application from Jupyter Notebook

from pyspark.sql import SparkSession

if __name__ == '__main__':
    spark = SparkSession.builder.appName('SparkApp').getOrCreate()

    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.feature import MinMaxScaler, VectorAssembler,VectorAssembler, MinMaxScaler
    from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col, lag, log, stddev, mean
    from pyspark.sql.functions import input_file_name,col, explode, last
    from pyspark.sql.session import SparkSession
    from pyspark.sql.types import DoubleType
    from pyspark.sql.window import Window
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns


    # Create SparkSession
    spark_session = SparkSession.builder\
        .appName("YourAppName")\
        .getOrCreate()
    
    print(f"This cluster relies on Spark '{spark_session.version}'")
    spark_session.sparkContext.setLogLevel("ERROR")
    spark_session.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.access.key", "s3access")
    spark_session.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.secret.key", "_s3access123$")
    spark_session.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.path.style.access", "true")
    spark_session.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    spark_session.sparkContext._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "http://localhost:9000")
    currency_raw_df = spark_session.read\
                                   .json("s3a://raw-currency/*.json")
    # Load JSON from MinIO (S3-compatible storage)
    currency_pairs = [field.name for field in currency_raw_df.schema.fields if field.name != "meta"]
    
    currency_dfs = {}
    
    for currency in currency_pairs:
        df = currency_raw_df.select(
            explode(col(f"{currency}.values")).alias(currency)
        ).select(
            col(f"{currency}.datetime").alias("date_time"),
            col(f"{currency}.open").cast(DoubleType()).alias(f"{currency}_open"),
            col(f"{currency}.high").cast(DoubleType()).alias(f"{currency}_high"),
            col(f"{currency}.low").cast(DoubleType()).alias(f"{currency}_low"),
            col(f"{currency}.close").cast(DoubleType()).alias(f"{currency}_close")
        )
        currency_dfs[currency] = df
    
    final_df = list(currency_dfs.values())[0]
    
    for currency, df in list(currency_dfs.items())[1:]:
        final_df = final_df.join(df, "date_time", "outer")
    
    final_pandas_df = final_df.toPandas()
    
    
    # Drop duplicate records based on date_time
    final_df_cleaned = final_df.dropDuplicates(["date_time"])
    
    final_df_cleaned.limit(10).toPandas()
    # Define a window specification ordered by date_time
    window_spec = Window.orderBy("date_time")
    
    # Apply forward fill for each currency column
    for col_name in final_df_cleaned.columns:
        if col_name != "date_time":
            final_df_cleaned = final_df_cleaned.withColumn(
                col_name, last(col_name, ignorenulls=True).over(window_spec)
            )
    
    numeric_cols = [col for col in final_df_cleaned.columns if col != "date_time"]
    
    mean_values = final_df_cleaned.agg(*[F.mean(c).alias(c) for c in numeric_cols]).collect()[0]
    
    mean_dict = {numeric_cols[i]: mean_values[i] for i in range(len(numeric_cols))}
    
    final_df_cleaned = final_df_cleaned.fillna(mean_dict)
    
    final_df_cleaned = final_df_cleaned.dropna(subset=[col for col in final_df_cleaned.columns if col != "date_time"])
    
    final_df_cleaned.limit(10).toPandas()
    df_summary = final_df_cleaned.describe().toPandas()
    
    df_summary
    
    rolling_window = Window.orderBy("date_time").rowsBetween(-4, 0)
    
    for col_name in numeric_cols:
        final_df_cleaned = final_df_cleaned.withColumn(
            f"{col_name}_SMA",
            mean(col_name).over(rolling_window)
        )
    
        final_df_cleaned = final_df_cleaned.withColumn(
            f"{col_name}_Volatility",
            stddev(col_name).over(rolling_window)
        )
    
        final_df_cleaned = final_df_cleaned.withColumn(
            f"{col_name}_Log_Return",
            log(col(col_name) / lag(col(col_name), 1).over(Window.orderBy("date_time")))
        )
        
    final_df_cleaned = final_df_cleaned.fillna(0, subset=[f"{col}_SMA" for col in numeric_cols])
    final_df_cleaned = final_df_cleaned.fillna(0, subset=[f"{col}_Volatility" for col in numeric_cols])
    
    final_df_cleaned = final_df_cleaned.fillna(0, subset=[f"{col}_Log_Return" for col in numeric_cols])
    
    feature_cols = [col for col in final_df_cleaned.columns if col not in ["date_time"]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    df_vectorized = assembler.transform(final_df_cleaned)
    
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    scaler_model = scaler.fit(df_vectorized)
    df_scaled = scaler_model.transform(df_vectorized)
    
    feature_cols = [col for col in final_df_cleaned.columns if col not in ["date_time", "USD/EUR_close"]]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features")
    
    train_ratio = 0.8
    total_rows = final_df_cleaned.count()
    split_index = int(total_rows * train_ratio)
    
    train_df = final_df_cleaned.limit(split_index)
    test_df = final_df_cleaned.subtract(train_df)
    
    print(f"Train size: {train_df.count()}, Test size: {test_df.count()}")
    
    lr = LinearRegression(featuresCol="scaled_features", labelCol="USD/EUR_close")
    rf = RandomForestRegressor(featuresCol="scaled_features", labelCol="USD/EUR_close", numTrees=50)
    gbt = GBTRegressor(featuresCol="scaled_features", labelCol="USD/EUR_close", maxIter=50)
    
    lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
    rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
    gbt_pipeline = Pipeline(stages=[assembler, scaler, gbt])
    
    lr_model = lr_pipeline.fit(train_df)
    rf_model = rf_pipeline.fit(train_df)
    gbt_model = gbt_pipeline.fit(train_df)
    
    predictions_lr = lr_model.transform(test_df)
    predictions_rf = rf_model.transform(test_df)
    predictions_gbt = gbt_model.transform(test_df)
    
    mse_evaluator = RegressionEvaluator(labelCol="USD/EUR_close", predictionCol="prediction", metricName="mse")
    mae_evaluator = RegressionEvaluator(labelCol="USD/EUR_close", predictionCol="prediction", metricName="mae")
    
    evaluation_results = {
        "Linear Regression": {
            "MSE": mse_evaluator.evaluate(predictions_lr),
            "MAE": mae_evaluator.evaluate(predictions_lr)
        },
        "Random Forest": {
            "MSE": mse_evaluator.evaluate(predictions_rf),
            "MAE": mae_evaluator.evaluate(predictions_rf)
        },
        "Gradient Boosted Trees": {
            "MSE": mse_evaluator.evaluate(predictions_gbt),
            "MAE": mae_evaluator.evaluate(predictions_gbt)
        }
    }
    
    for model, metrics in evaluation_results.items():
        print(f"{model}: MSE = {metrics['MSE']}, MAE = {metrics['MAE']}")
    
    best_model = min(evaluation_results, key=lambda x: evaluation_results[x]["MSE"])
    
    print(f"Best Model: {best_model}")
    
    lr_model.save("s3a://models/linear_regression")
    rf_model.save("s3a://models/random_forest")
    gbt_model.save("s3a://models/gradient_boosted_tree")
    