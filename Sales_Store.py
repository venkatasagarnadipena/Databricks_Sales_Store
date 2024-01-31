# Databricks notebook source
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import *

# COMMAND ----------

df=spark.read.format("csv").option("header",True).option("delimiter",",").option("inferschema",True).load("dbfs:/FileStore/tables/Sales/SalesStore")
df.show(5,False)

# COMMAND ----------

# MAGIC %md #Data Manuliplation Task

# COMMAND ----------

# 1. Order_Month:
#     Derivation: Extract the month from the Order_Date column.
#     Example: If Order_Date is '2021-02-15', the derived column would be 'February'.
df=df.withColumn("Order_Month",date_format(df.Order_Date,"MMMM"))
df.show(5,False)

# COMMAND ----------

# 2. Shipping Duration:
#     Derivation: Calculate the duration between Order_Date and Ship_Date.
#     Example: If Order_Date is '2021-02-15' and Ship_Date is '2021-02-20', the derived column could be '5 days'.
# df=df.withColumn("Shipping_Duration",concat_ws(datediff(df.Ship_Date,df.Order_Date),lit("Days")))
df = df.withColumn("Shipping_Duration", concat(datediff(df.Ship_Date, df.Order_Date),lit(" Days")))
df.show(5,False)

# COMMAND ----------

# 3. Sales per Quantity:
#     Derivation: Calculate the ratio of Sales to Quantity.
#     Example: If Sales is 500 and Quantity is 2, the derived column would be 250 (Sales per Quantity)
df=df.withColumn("Sales_Per_Quantity",round((df.Sales/df.Quantity),2))
df.show(5,False)

# COMMAND ----------

# 4. Profit Margin:
#     Derivation: Calculate the percentage profit margin based on Profit and Sales.
#     Example: If Profit is 10 and Sales is 500, the derived column could be 2% (Profit Margin).
df=df.withColumn("Profit_Margin",round((df.Profit/df.Sales)*100,2))
df.show(5,False)

# COMMAND ----------

# 5. Year from Order Date:
#     Derivation: Extract the year from the Order_Date column.
#     Example: If Order_Date is '2021-02-15', the derived column would be '2021'.

df=df.withColumn("Year_from_Order_Date",date_format(df.Order_Date,"yyyy"))
df.show(5,False)

# COMMAND ----------

# 8. Order Priority:
#     Derivation: Assign priority levels based on the Ship_Mode column (e.g., 'High', 'Medium', 'Low').
#     Example: If Ship_Mode is 'Air', the derived column could be 'High'.
df=df.withColumn("Order_Priority",when(col("Ship_Mode")=="Air","High").when(df.Ship_Mode=="Ground","Low").otherwise("Medium"))
df.show(5,False)

# COMMAND ----------

# 10. Order Value:
#     Derivation: Calculate the total value of each order by multiplying Sales and Quantity.
#     Example: If Sales is 200 and Quantity is 3, the derived column would be 600 (Order Value).
df=df.withColumn("Order_Value",df.Sales*df.Quantity)
df.show(5,False)

# COMMAND ----------

# Change the region value of all the Midwest to west
df=df.withColumn("Region",when(col("Region")=="Midwest","West").otherwise(col("Region")))
df.show(5,False)
df.createOrReplaceTempView("a_df")

# COMMAND ----------

# MAGIC %md # Data Analysis 

# COMMAND ----------

# Question: Rank orders based on profit margin in descending order
a_df=df.select("Sub_Category","Profit_Margin").withColumn("Rank_Of_Orders",dense_rank().over(Window.orderBy(col("Profit_Margin").desc()))).distinct()
a_df.show(5,False)

# COMMAND ----------

# MAGIC %sql select Sub_Category,Profit_Margin,dense_rank() over( Order by Profit_Margin desc) as Rank_Of_Orders  from a_df

# COMMAND ----------

# Question: Find orders with the same Order_Date, Ship_Date, Product_Reference, and State.
a_df=df.select("Order_Date","Ship_Date","Product_Reference","State")
a_df=a_df.withColumn("Same_Orders",row_number().over(Window.partitionBy(df.Order_Date,df.Ship_Date,df.Product_Reference,df.State).orderBy(df.Order_Date,df.Ship_Date,df.Product_Reference,df.State)))
a_df=a_df.filter(a_df.Same_Orders>=2).drop("Same_Orders").distinct()
a_df.show(5,False)

# COMMAND ----------

# MAGIC %sql select Order_Date, Ship_Date, Product_Reference,State from (select Order_Date, Ship_Date, Product_Reference,State,count(*) from a_df
# MAGIC group by 1,2,3,4
# MAGIC having count(*)>1)

# COMMAND ----------

# Question: Retrieve the latest order for each unique Product_Reference.
a_df=df.select("Product_Reference","Order_Date").groupBy("Product_Reference").agg(max(col("Order_Date")).alias("Latest_orderDate"))
a_df.show(5,False)

# COMMAND ----------

# MAGIC %sql select Product_Reference,max(Order_Date) as Latest_orderDate from a_df
# MAGIC group by 1

# COMMAND ----------

# Question: Calculate the cumulative sales for each month, ordered by Order_Date.
a_df=df.select("Order_Date","Order_Month","Sales").withColumn("Rolling_Sum_By_Month",sum(col("Sales")).over(Window.partitionBy(df.Order_Date,df.Order_Month).orderBy(df.Order_Date.asc()))).drop("Sales").distinct()
a_df.show(5,False)

# COMMAND ----------

# MAGIC %sql select distinct Order_Date,Order_Month,sum(Sales) over(partition by order_date,Order_Month order by order_Date Asc) Rolling_Sum_By_Month from a_df

# COMMAND ----------

# Question: Determine the average sales for each month and identify any seasonal trends.
a_df=df.select("Order_Date","Order_Month","Sales").groupBy(df.Order_Date,df.Order_Month).agg(avg(col("Sales")).alias("Avg_Sales_Per_Month"))
a_df.show(5,False)

# COMMAND ----------

# MAGIC %sql select Order_Date,Order_Month,avg(Sales) Avg_Sales_Per_Month from a_df 
# MAGIC group by 1,2

# COMMAND ----------

# Question: Find products (based on State) with the highest cumulative order values.
a_df=df.select("Sub_Category","State","Order_Value").groupBy(col("State")).agg(max(col("Order_Value")).alias("Highest_Order_Value"))
a_df.show(5,False)

# COMMAND ----------

# MAGIC %sql select State,max(Order_Value) from a_df
# MAGIC group by 1

# COMMAND ----------

# Question: Analyze and identify what mode is the most prefered way to transport.
a_df=df.select("Ship_Mode","Order_Priority").groupBy(col("Ship_Mode")).agg(count(col("Order_Priority")).alias("Preffered_mode"))
a_df.show(5,False)

# COMMAND ----------

# MAGIC %sql select Ship_Mode,count(Order_Priority) Preffered_mode from a_df
# MAGIC group by 1

# COMMAND ----------

# MAGIC %md #only Pyspark

# COMMAND ----------

# Question: Identify products where the profit margin has been decreasing over time.
a_df=df.select("Order_Date","Sub_Category","Profit_Margin").withColumn("Lag",lag("Profit_Margin").over(Window.partitionBy(col("Sub_Category")).orderBy(col("Order_Date").asc())))
a_df=a_df.select("Sub_Category").filter((col("Profit_Margin")>col("Lag")))
a_df.show(5,False)

# COMMAND ----------

# Question: Calculate the average profit margin for each region.
a_df=df.select("Region","Profit_Margin").groupBy(col("Region")).agg(round(avg(col("Profit_Margin")),2).alias("Profit_By_Region"))
a_df.show(5,False)
