# SQP train

## Extract SQP pattern from train data
- run sqppatterns with action `prepare-query` starting from train data as shown in the help:
    ```
    sqppatterns.py --help
    ```
    This will create a file containing queries
- run sparql2graph app (based on apache jena) to parse the query file.
    To build it run
    ```
    mvn assembly:assembly
    ```
    then run it as shown in the help:
    ```
    java -cp target/sparql2graph-1.0-SNAPSHOT-jar-with-dependencies.jar com.mycompany.app.App --help
    ```
    At this point you should have a file containing query graphs.
- run sqppatterns to detect patterns starting from graph file using
  `get-pattern` action as shown in the help:
    ```
    sqppatterns.py --help
    ```
    Now you should have a csv file ready for the model containing structural
    query patterns too.
## Model Train
- open the notebook `ModelTrain.ipynb` (tested and used in Google Colab) and run
  it to train the model and then export it.
## Test
- repeat the steps from "Extract SQP pattern from train data" section with test data instead of train in order to
  evaluate model performace.
## Model
- get the model file (its extension is `.h5`)
- put the model in the correct folder so that it can be used by `kgqalib`.