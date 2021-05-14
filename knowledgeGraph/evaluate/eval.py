#!/usr/bin/env python
import click
import json
import pandas as pd
import numpy as np

def patterns(df):
    """
    Calculates metrics about pattern prediction
    """
    # confusion matrix
    tots = df['patterns'].value_counts()
    idx = sorted(tots.index)
    cols = sorted(list(df['predicted_pattern'].value_counts().index) + ['nan', 'tot'])
    tot = df['patterns'].shape[0]
    cm = pd.DataFrame(np.zeros((len(idx), len(cols))),
        index = idx,
        columns=cols)
    cm.index.name = 'gold'
    for i, row in df[['patterns', 'predicted_pattern']].iterrows():
        y = row['predicted_pattern']
        if type(y) != str and np.isnan(y):
            y = 'nan'
        cm.loc[row['patterns'], y] += 1
    cm['tot'] = tots

    accuracy = np.diag(cm[idx]).sum()/tot

    normalized_cm = cm[sorted(set(cols) - {'tot'})].div(cm['tot'], axis=0)

    display = """{}
Pattern prediction performance:
Accuracy: {}
Normalized Confusion matrix:
{}
Confusion matrix:
{}
""".format('-'*30, accuracy, normalized_cm.to_markdown(), cm.to_markdown())
    return display, accuracy, normalized_cm, cm

def to_jsonl(df, outpath):
    with open(outpath, 'w') as fd:
        for i, row in df.iterrows():
            line = json.dumps({"index": i, "query": row["predicted_query"]}) + "\n"
            fd.write(line)

@click.command()
@click.option('--action', type=click.Choice(['eval', 'prepare-queries', 'merge-graphs'], case_sensitive=False), help='Main action')
@click.option('-o', '--output-file', type=click.STRING, help='Prepare output path', default=None)
@click.option('-g', '--graphs-input-file', type=click.STRING, help='File containing predicted queries graphs created from sparql2graph jena app', default=None)
@click.argument('input', type=click.Path(exists=True))
def main(action, output_file, graphs_input_file, input):
    """
    Given as input the file created with `predictQueries`, it calculates performance metrics.
    Actions:
        eval: evaluates given the input file containing predicted graphs
        prepare-queries: prepares the predicted queries creating the output file so that jena sparql2graph app can create the graphs
        merge-graphs: given the input jsonl file and the graph for the predicted queries produced by jena sparql2graph app it creates
            a complete jsonl file as output. Ready for eval
    """
    # load jsonl
    def getLines(fname):
        with open(fname, 'r') as fd:
            for line in fd.readlines():
                yield json.loads(line)

    df = pd.DataFrame(getLines('out.jsonl'))
    if action == 'eval':
        print(patterns(df)[0])
    elif action == 'prepare-queries':
        print(f"Writing queries to {output_file} for apache jena...")
        to_jsonl(df.loc[df["predicted_query"].notna(), ["predicted_query"]], output_file)

if __name__ == '__main__':
    main()