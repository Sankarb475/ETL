# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 12:46:39 2019

@author: sankar biswas
"""

import pandas as pd
from cassandra.cluster import Cluster
import pyspark
from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, coalesce
import mysql.connector
import numpy as np
import math 
from neo4j import GraphDatabase
from neo4j.v1 import GraphDatabase, basic_auth

def handlingNA(a):
    for i in range(len(a)):
        if a[i]== None or a[i].lower() == "n/a" or a[i] == '':
            a[i] = np.NaN
    return a

def metadataCapturing(option, name, rows):
    driver = GraphDatabase.driver("bolt://localhost", auth=basic_auth(user = "neo4j", password = "1097"))
    session = driver.session()
    query1 = """MERGE (N:Entity{name:"%s"})
                SET N.Age = %d
                RETURN N"""%(name,rows)
    session.run(query1)
    if option == "MySQL":
        query2 = """MATCH (P:Entity{name:"Cassandra"})
                    MATCH (N:Entity{name:"MySQL"})
                    MERGE (P)-[:Success]->(N)"""
        session.run(query2)
    session.close()
    
# Fetching raw data from Cassandra
def Cassandra_integration():
    cluster = Cluster(['10.196.104.21'])
    session = cluster.connect('project')
    query = "select * from housingdata"
    df = pd.DataFrame(list(session.execute(query)))
    
    queryCount = "select count(*) from housingdata"
    count = pd.DataFrame(list(session.execute(queryCount)))
    metadataCapturing("Cassandra", "Cassandra", count["count"].tolist()[0])
    
    # filtering out the unnecessary columns
    df = df.rename(columns = {"postalcode":"postal_code", "from_price":"price_from","to_price":"price_to",
                              "propertytype":"property_type","propertyfeatures":"property_features","noofbeds":"beds"
                             ,"noofbaths":"baths","parkingspace":"parking","nameofproperty":"name_of_property"})
    df = df.loc[:, ["postal_code", "price_from", "price_to", "price", "property_type", "property_features", "beds", "baths", "parking", "name_of_property"]]


    df['property_type'] = df['property_type'].apply(lambda x: 'Rural' if x==None or x.lower() == "n/a" else x)
    
    df = df.apply(lambda x: handlingNA(x.tolist()))
    
    df = df.drop_duplicates()
    
    # dropping the rows which has any "Na" values 
    df = df.dropna(subset=["postal_code", "price_from", "price_to", "beds", "baths", "parking", "name_of_property","property_features"])
    
    df['Region'] = df["name_of_property"].str.split(',').str[-1]
    
    df['Region'] = df['Region'].str.strip()
    
    df = df.drop(["name_of_property"], axis = 1)
    
    df['Area'] = df['Region'].str.split(" ").str[:-2]
    
    df['Area'] = df['Area'].apply(' '.join)
    
    df["State"] = df["Region"].str.split(" ").str[-2]
    
    df = df.drop(['Region'], axis =1)
    
    return df

# handling NaN values
def isNaN(num):
    return num != num

# removing white spaces from a list of strings
def listStrip(a):
    for i in range(len(a)):
        a[i] = float(a[i].strip())
    return a
 
#extracting price column values after cleansing
def priceExtraction(a):
    if isNaN(a):
        return a
    a = str(a).lower().strip()
    a = a.replace("$", "").replace("million", "").replace("M","").replace("m","").replace(",","").strip()
    
    #for values like : "539000 to 570000"
    
    if "to" in a.lower():                 
        mm = listStrip(a.split("to"))
        price = (mm[0] + mm[1])/2
        if price < 10:
            return (price*1000000)
        return price
    
    #for values like : "539000 1" | "10000 000"
    
    if len(a.split(" "))>1:
        return float("NaN")
    
    #for values like 1.1875 
    a = float(a)
    if a < 10:
        return (a*1000000)
    else:    
        return a 

# ETL on beds, bath and parking columns
def getPrice():
    df = Cassandra_integration()
    df["Price123"] = df["price"].apply(lambda x: priceExtraction(x))
    df = df.drop(['price'], axis =1)
    
    df['beds'] = df['beds'].str.strip()

    df['beds'] = df['beds'].str.split(' ').str[0]

    df['baths'] = df['baths'].str.strip()

    df['baths'] = df['baths'].str.split(' ').str[0]

    df['parking'] = df['parking'].str.strip()

    df['parking'] = df['parking'].str.split(' ').str[0]

    df['property_features'] = df['property_features'].str.strip().str[:-2]
    
    #print(df)
    return df

# removing duplicate features from feature list of the raw data
def func(a):
    b = []
    c = []
    for i in range(len(a)):
        if len(a[i])==0:
            continue
        c.append(a[i].strip().lower())      
    a = list(set(c))
    return a

#feature weight assignment
def featureExtraction(a):
    featured_dict = ["air conditioning", "gym", "swimming pool", "wardrobe", "city view", "dishwasher", 
                 "double glazed windows", "floorboards", "heating", "intercom", "north facing",
                "close transport hubs", "close to schools", "warehouse", "balcony / deck", "cable or satellite",
                "internal laundry", "world class shops", "garden / courtyard", "alarm system"]

    featured_weight = {"air conditioning":10 , "gym":5, "swimming pool":5, "wardrobe":2, "city view":4, "dishwasher":2, 
                 "double glazed windows":3, "floorboards":4, "heating":5, "intercom":2, "north facing":2,
                "close transport hubs":5, "close to schools" :10, "warehouse":3, "balcony / deck":5, "cable or satellite":5,
                "internal laundry":10, "world class shops":5, "garden / courtyard":5, "alarm system":8}
    count = 0
    for i in a:
        if i.lower() in featured_dict:
            count = count + featured_weight[i.lower()]
    return count

# converting string into integer 
def toInt(a):
    a = a.replace("$", "").replace(",","")
    return int(a)

# feature weight implementation
def get_features():
    df1 = getPrice()

    df1["featureList"] = df1["property_features"].str.split(",")

    df1["disFeature"] = df1["featureList"].apply(lambda x: func(x))

    df1 = df1.drop(['featureList', 'property_features'], axis =1)
    
    df1["countOfFeatures"] = df1["disFeature"].apply(lambda x: featureExtraction(x))
    
    df1 = df1.drop(["disFeature"], axis = 1)
    
    #type_casting 
    df1 = df1[df1["price_to"] != 'Any']
    df1["price_from"] = df1["price_from"].apply(lambda x: toInt(x))
    df1["price_to"] = df1["price_to"].apply(lambda x: toInt(x))

    df1.to_csv(r"C:\Users\sbiswas149\Applications\Cassandra\data.csv")
    
    #print(df1)
    return df1

#pyspark ETL, filling up empty price values using median of the range
def pyspark():
    conf = SparkConf().setAppName("PySparkApp").setMaster("local")
    #conf = SparkConf()
    sc = SparkContext(conf = conf)

    #spark = SparkSession.builder.appName("WordCount").master("local").config(conf = conf).getOrCreate()
    sqlContext = SQLContext(sc)
    
    get_features()
    
    sdf = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load(r"C:\Users\sbiswas149\Applications\Cassandra\data.csv")

    ops1 = "(price_from + price_to)/2"
    data = sdf.withColumn("MedianPrice", expr(ops1))

    tmp = data.withColumn('final_price', coalesce(data['Price123'],data['MedianPrice']))

    finaldata = tmp.drop("price", "disFeature")

    finaldataPD = finaldata.toPandas()
    
    finaldataPD['price_to'] = finaldataPD['price_to'].astype(str).astype(float)

    finaldataPD['Price123'] = finaldataPD['Price123'].astype(str).astype(float)

    finaldataPD['beds'] = finaldataPD['beds'].astype(str).astype(int)

    finaldataPD['baths'] = finaldataPD['baths'].astype(str).astype(int)

    finaldataPD['parking'] = finaldataPD['parking'].astype(str).astype(int)

    df123 = finaldataPD.copy()

    df123 = df123.replace({pd.np.nan: None})
    sc.stop()
    
    return df123


#writing the processed data into Mysql
def WritingToMysql():    
    df123 = pyspark()
    query1 = ("INSERT INTO visualize.test_data "
              "(postal_code,price_from,price_to,property_type,beds,baths,parking, region, state, features) "
              "VALUES (%s, %s, %s, %s,%s, "
              "%s, %s, %s, %s, %s)")

    dataMysql = df123[["postal_code","price_from","price_to","property_type","beds","baths","parking","Area", "State","countOfFeatures"]].values.tolist()

    filedata = df123[["postal_code","price_from","price_to","property_type","beds","baths","parking","Area", "State","countOfFeatures"]]
    filedata.to_csv(r"C:\Users\sbiswas149\Applications\Cassandra\MLdata.csv")
    
    cnx = mysql.connector.connect(user='root', password='admin123',
                              host='10.196.104.221',
                              database='visualize')
    """
    
    cnx = mysql.connector.connect(user='root', password='1097',
                              host='127.0.0.1',
                              database='visualize')
    """
    cursor = cnx.cursor()

    for i in dataMysql:
        j = tuple(i)
        cursor.execute(query1, j)
        
    queryMCount = "select count(*) from visualize.test_data"
    rows = cursor.execute(queryMCount) 
    records = cursor.fetchall()
    dataM = pd.DataFrame(records)
    metadataCapturing("MySQL","MySQL", dataM[0].tolist()[0])
    
    cnx.commit()
    cursor.close()
    cnx.close()
 
# triggering the ETL application
if __name__ == '__main__':
    WritingToMysql()
    
  
