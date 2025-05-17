const codeData = [
    {
        title: "Create Table",
        description: "Learn to create structured tables before manipulating data.",
        code: `
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("example").getOrCreate()
        df = spark.createDataFrame([(1, "Alice")], ["ID", "Name"])
        df.show()
        `,
        dependencies: [] // No prerequisites
    },
    {
        title: "Read File",
        description: "Before working with data, understand how to read files.",
        code: `
        df = spark.read.csv("data.csv", header=True, inferSchema=True)
        df.show()
        `,
        dependencies: ["Create Table"] // Requires table knowledge first
    },
    {
        title: "DataFrame Transformations",
        description: "Perform operations like filtering, joining, and aggregations.",
        code: `
        df_filtered = df.filter(df["ID"] > 1)
        df_selected = df.select("Name")
        `,
        dependencies: ["Read File"] // Requires file reading first
    },
    {
        title: "DataFrame Basics",
        usage: 150,
        description: "Create and view data in PySpark.",
        code: `
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.appName("example").getOrCreate()
        data = [(1, "Alice"), (2, "Bob")]
        df = spark.createDataFrame(data, ["ID", "Name"])
        df.show()
        `
    },
    {
        title: "DataFrame Transformations",
        usage: 130,
        description: "Filter and select columns in a PySpark DataFrame.",
        code: `
        df_filtered = df.filter(df["ID"] > 1)
        df_selected = df.select("Name")
        `
    },
    {
        title: "Joins",
        usage: 120,
        description: "Join two DataFrames in PySpark.",
        code: `
        df1 = spark.createDataFrame([(1, "Alice")], ["ID", "Name"])
        df2 = spark.createDataFrame([(1, "USA")], ["ID", "Country"])

        df_joined = df1.join(df2, "ID", "inner")
        df_joined.show()
        `
    }
];

export default codeData;
