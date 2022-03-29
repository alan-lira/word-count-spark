from configparser import ConfigParser
from pathlib import Path
from pyspark import RDD, SparkConf, SparkContext
from pyspark.sql import SparkSession
from sys import argv
from time import time
from typing import List


def check_if_has_valid_number_of_arguments(argv_list: list) -> None:
    number_of_arguments_expected = 2
    arguments_expected_list = ["word_counter_config_file", "spark_application_submission_settings_file"]
    number_of_arguments_provided = len(argv_list) - 1
    if number_of_arguments_provided != number_of_arguments_expected:
        number_of_arguments_expected_message = \
            "".join([str(number_of_arguments_expected),
                     " arguments were" if number_of_arguments_expected > 1 else " argument was"])
        number_of_arguments_provided_message = \
            "".join([str(number_of_arguments_provided),
                     " arguments were" if number_of_arguments_provided > 1 else " argument was"])
        invalid_number_of_arguments_message = \
            "Invalid number of arguments provided!\n" \
            "{0} expected: {1}\n" \
            "{2} provided: {3}".format(number_of_arguments_expected_message,
                                       ", ".join(arguments_expected_list),
                                       number_of_arguments_provided_message,
                                       ", ".join(argv_list[1:]))
        raise ValueError(invalid_number_of_arguments_message)


def check_if_file_exists(file_path: Path) -> None:
    if not file_path.is_file():
        file_not_found_message = "'{0}' not found. The application will halt!".format(str(file_path))
        raise FileNotFoundError(file_not_found_message)


def parse_word_counter_config(word_counter_config_file: Path) -> List:
    config_parser = ConfigParser()
    config_parser.optionxform = str
    config_parser.read(word_counter_config_file,
                       encoding="utf-8")
    input_file_path = config_parser.get("Input Settings",
                                        "input_file")
    output_directory_path = config_parser.get("Output Settings",
                                              "output_directory")
    number_of_partitions_for_rdd = int(config_parser.get("General Settings",
                                                         "number_of_partitions_for_rdd"))
    counter_config = [input_file_path, output_directory_path, number_of_partitions_for_rdd]
    return counter_config


def parse_spark_application_submission_settings(spark_application_submission_settings_file: Path) -> List:
    config_parser = ConfigParser()
    config_parser.optionxform = str
    config_parser.read(spark_application_submission_settings_file,
                       encoding="utf-8")
    spark_application_submission_settings = \
        list(config_parser.items("Spark Application Submission Settings (SparkConf)"))
    return spark_application_submission_settings


def create_spark_conf(spark_application_properties: List) -> SparkConf:
    spark_conf = SparkConf()
    for (key, value) in spark_application_properties:
        spark_conf.set(key, value)
    return spark_conf


def get_or_create_spark_session(spark_conf: SparkConf) -> SparkSession:
    spark_session = \
        SparkSession \
        .builder \
        .config(conf=spark_conf) \
        .getOrCreate()
    return spark_session


def get_spark_context_from_spark_session(spark_session: SparkSession) -> SparkContext:
    spark_context = spark_session.sparkContext
    return spark_context


def repartition_rdd(rdd: RDD,
                    new_number_of_partitions: int) -> RDD:
    current_number_of_partitions = rdd.getNumPartitions()
    if current_number_of_partitions > new_number_of_partitions:
        # Execute Coalesce (Spark Less-Wide-Shuffle Transformation) Function
        rdd = rdd.coalesce(new_number_of_partitions)
    if current_number_of_partitions < new_number_of_partitions:
        # Execute Repartition (Spark Wider-Shuffle Transformation) Function
        rdd = rdd.repartition(new_number_of_partitions)
    return rdd


def execute_word_count(spark_context: SparkContext,
                       word_counter_config: List) -> None:
    # Get Input File Path
    input_file_path = word_counter_config[0]
    # Get Output Directory Path
    output_directory_path = word_counter_config[1]
    # Get Number of Partitions For RDD
    number_of_partitions_for_rdd = word_counter_config[2]
    # Step 1. Read The Input Data Using The 'textFile' Function, Generating a RDD;
    rdd1 = spark_context.textFile(input_file_path)
    # Set The Desired Number Of Partitions;
    rdd1 = repartition_rdd(rdd1, number_of_partitions_for_rdd)
    # Step 2. Generate a RDD Containing Lists of Words Using The 'flatMap' Function;
    rdd2 = rdd1.flatMap(lambda line: line.split(" "))
    # Step 3. Generate a RDD Containing Tuples <key, value> = <word, number_of_occurrences=1> Using The 'map' Function;
    rdd3 = rdd2.map(lambda word: (word, 1))
    # Step 4. Group Tuples By word And Sum Their Corresponding number_of_occurrences Using The 'reduceByKey' Function;
    rdd4 = rdd3.reduceByKey(lambda a, b: a + b)
    # Step 5. Write The Resulting RDD as Text File Using The 'saveAsTextFile' Function.
    rdd4.saveAsTextFile(output_directory_path)


def stop_spark_session(spark_session: SparkSession) -> None:
    spark_session.stop()


def count(argv_list: list) -> None:
    # Begin
    begin_time = time()
    # Print Application Start Notice
    print("Application Started!")
    # Check if Has Valid Number of Arguments
    check_if_has_valid_number_of_arguments(argv_list)
    # Read Word Counter Config File
    word_counter_config_file = Path(argv_list[1])
    # Check If Word Counter Config File Exists
    check_if_file_exists(word_counter_config_file)
    # Parse Word Counter Config
    word_counter_config = parse_word_counter_config(word_counter_config_file)
    # Read Spark Application Submission Settings File
    spark_application_submission_settings_file = Path(argv_list[2])
    # Check If Spark Application Submission Settings File Exists
    check_if_file_exists(spark_application_submission_settings_file)
    # Parse Spark Application Submission Settings
    spark_application_submission_settings = \
        parse_spark_application_submission_settings(spark_application_submission_settings_file)
    # Create SparkConf
    spark_conf = create_spark_conf(spark_application_submission_settings)
    # Get or Create Spark Session
    spark_session = get_or_create_spark_session(spark_conf)
    # Get Spark Context
    spark_context = get_spark_context_from_spark_session(spark_session)
    # Execute Word Count
    execute_word_count(spark_context,
                       word_counter_config)
    # Stop Spark Session
    stop_spark_session(spark_session)
    # Print Application End Notice
    print("Application Finished Successfully!")
    end_time = time()
    print("Duration Time: {0} seconds.".format(end_time - begin_time))
    # End
    exit(0)


if __name__ == "__main__":
    count(argv)
