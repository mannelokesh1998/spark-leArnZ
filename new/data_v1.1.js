const codeData = [
    {
        title: "Create Table",
        description: "Start by making a simple table before working with data.",
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
        description: "Load a CSV file into PySpark to begin processing.",
        code: `
        df = spark.read.csv("data.csv", header=True, inferSchema=True)
        df.show()
        `,
        dependencies: ["Create Table"] // Requires table knowledge first
    },
    {
        title: "Filter Data",
        description: "Keep only rows where the condition is met.",
        code: `
        df_filtered = df.filter(df["ID"] > 1)
        df_filtered.show()
        `,
        dependencies: ["Read File"] // Requires file reading first
    },
    {
        title: "Select Columns",
        description: "Pick specific columns instead of the whole table.",
        code: `
        df_selected = df.select("Name")
        df_selected.show()
        `,
        dependencies: ["Read File"] // Requires file reading first
    },
    {
        title: "INNER JOIN",
        description: "Merge two tables where common data exists.",
        code: `
        df1: ID, Name  →  [1, "Alice"]
        df2: ID, Country  →  [1, "USA"]

        INNER JOIN (df1, df2) → OUTPUT: [1, "Alice", "USA"]
        `,
        dependencies: ["Create Table"]
    },
    {
        title: "LEFT JOIN",
        description: "Keep everything from the first table, match when possible.",
        code: `
        df1: ID, Name  →  [1, "Alice"]
        df2: ID, Country  →  [1, "USA"], [2, "India"]

        LEFT JOIN (df1, df2) → OUTPUT: [1, "Alice", "USA"], [2, "Bob", NULL]
        `,
        dependencies: ["Create Table"]
    },
    {
        title: "RIGHT JOIN",
        description: "Keep everything from the second table, match when possible.",
        code: `
        df1: ID, Name  →  [1, "Alice"]
        df2: ID, Country  →  [1, "USA"], [2, "India"]

        RIGHT JOIN (df1, df2) → OUTPUT: [1, "Alice", "USA"], [2, NULL, "India"]
        `,
        dependencies: ["Create Table"]
    },
    {
        title: "Group By",
        description: "Summarize data based on categories.",
        code: `
        df.groupBy("Country").count().show()
        `,
        dependencies: ["Read File"]
    },
    {
        title: "Sort Data",
        description: "Arrange data in ascending or descending order.",
        code: `
        df_sorted = df.orderBy(df["Name"].asc())
        df_sorted.show()
        `,
        dependencies: ["Read File"]
    }
];

export default codeData;
