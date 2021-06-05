#!/usr/bin/env python
import click
import json
import pandas as pd
import numpy as np
import networkx as nx
import sys
import statistics

def graph2nx(graphdict):
    if graphdict is None:
        return None
    if type(graphdict) == str:
        graphdict = json.loads(graphdict)
    G = nx.from_dict_of_dicts(graphdict, create_using=nx.DiGraph)
    attributes = {n:n for n in G.nodes}
    nx.set_node_attributes(G, attributes, 'label')
    return G

# from networkx.algorithms.isomorphism
def categorical_node_match_strict(attr):
    if isinstance(attr, str):

        def match(data1, data2):
            if attr in data1 and attr in data2:
                val1 = data1.get(attr)
                val2 = data2.get(attr)
                if isinstance(val1, str) and isinstance(val2, str):
                    return val1 == val2
                else:
                    return False
            else:
                return False
    return match

# from networkx.algorithms.isomorphism
def categorical_node_match_permissive_strict(attr):
    if isinstance(attr, str):
        global categorical_node_match_permissive_vars_assignments
        global categorical_node_match_permissive_vars_assignments_inv
        categorical_node_match_permissive_vars_assignments = {} # val1 -> val2
        categorical_node_match_permissive_vars_assignments_inv = {} # val2 -> val1

        def match(data1, data2):
            if attr in data1 and attr in data2:
                val1 = data1.get(attr)
                val2 = data2.get(attr)
                if isinstance(val1, str) and isinstance(val2, str):
                    if val1 == val2:
                        # exact match
                        return True
                    else:
                        if val1[0] == '?' and val2[0] == '?':
                            # val1 or val2 or both are variables
                            # keep track of variable assignments
                            global categorical_node_match_permissive_vars_assignments
                            global categorical_node_match_permissive_vars_assignments_inv
                            if val1 in categorical_node_match_permissive_vars_assignments:
                                if val2 == categorical_node_match_permissive_vars_assignments.get(val1):
                                    # agreed with previous assignment
                                    return True
                                else:
                                    return False
                            elif val2 in categorical_node_match_permissive_vars_assignments_inv:
                                # val1 was not in categorical_node_match_permissive_vars_assignments
                                # -> non match
                                return False
                            else:
                                categorical_node_match_permissive_vars_assignments[val1] = val2
                                categorical_node_match_permissive_vars_assignments_inv[val2] = val1
                                return True
                        else:
                            return False
                else:
                    return False
            else:
                return False
    return match

# from networkx.algorithms.isomorphism
def categorical_node_match_permissive(attr):
    if isinstance(attr, str):
        global categorical_node_match_permissive_vars_assignments
        global categorical_node_match_permissive_vars_assignments_inv
        categorical_node_match_permissive_vars_assignments = {} # val1 -> val2
        categorical_node_match_permissive_vars_assignments_inv = {} # val2 -> val1

        def match(data1, data2):
            if attr in data1 and attr in data2:
                val1 = data1.get(attr)
                val2 = data2.get(attr)
                if isinstance(val1, str) and isinstance(val2, str):
                    if val1 == val2:
                        # exact match
                        return True
                    else:
                        if val1[0] == '?' or val2[0] == '?':
                            # val1 or val2 or both are variables
                            # keep track of variable assignments
                            global categorical_node_match_permissive_vars_assignments
                            global categorical_node_match_permissive_vars_assignments_inv
                            if val1 in categorical_node_match_permissive_vars_assignments:
                                if val2 == categorical_node_match_permissive_vars_assignments.get(val1):
                                    # agreed with previous assignment
                                    return True
                                else:
                                    return False
                            elif val2 in categorical_node_match_permissive_vars_assignments_inv:
                                # val1 was not in categorical_node_match_permissive_vars_assignments
                                # -> non match
                                return False
                            else:
                                categorical_node_match_permissive_vars_assignments[val1] = val2
                                categorical_node_match_permissive_vars_assignments_inv[val2] = val1
                                return True
                        else:
                            return False
                else:
                    return False
            else:
                return False
    return match

# from networkx.algorithms.isomorphism
def categorical_node_match_permissive_both_vars(attr):
    if isinstance(attr, str):

        def match(data1, data2):
            if attr in data1 and attr in data2:
                val1 = data1.get(attr)
                val2 = data2.get(attr)
                if isinstance(val1, str) and isinstance(val2, str):
                    if val1 == val2:
                        # exact match
                        return True
                    else:
                        if val1[0] == '?' and val2[0] == '?':
                                return True
                        else:
                            return False
                else:
                    return False
            else:
                return False
    return match

# from networkx.algorithms.isomorphism
def categorical_node_match_permissive_at_least_one_var(attr):
    if isinstance(attr, str):

        def match(data1, data2):
            if attr in data1 and attr in data2:
                val1 = data1.get(attr)
                val2 = data2.get(attr)
                if isinstance(val1, str) and isinstance(val2, str):
                    if val1 == val2:
                        # exact match
                        return True
                    else:
                        if val1[0] == '?' or val2[0] == '?':
                                return True
                        else:
                            return False
                else:
                    return False
            else:
                return False
    return match

def isom(df):
    total = df.shape[0]
    correct = {
        'iso_only': 0,
        'iso_strict': 0,
        'iso_perm_strict': 0,
        'iso_perm': 0,
        'iso_perm_at_least_one_var': 0,
        'iso_perm_both_vars': 0,
        'total': total
    }
    for _, row in df.iterrows():
        G1 = graph2nx(row['graph'])
        G2 = graph2nx(row['predicted_graph'])
        if G1 is None or G2 is None:
            continue
        else:
            isom = nx.is_isomorphic(G1, G2)
            if isom:
                correct['iso_only'] += 1

            iso_strict = nx.is_isomorphic(G1, G2,
                node_match=categorical_node_match_strict("label"),
                edge_match=categorical_node_match_strict("label"))
            if iso_strict:
                correct['iso_strict'] += 1

            iso_perm_strict = nx.is_isomorphic(G1, G2,
                node_match=categorical_node_match_permissive_strict("label"),
                edge_match=categorical_node_match_strict("label"))
            if iso_perm_strict:
                correct['iso_perm_strict'] += 1

            iso_perm = nx.is_isomorphic(G1, G2,
                node_match=categorical_node_match_permissive("label"),
                edge_match=categorical_node_match_strict("label"))
            if iso_perm:
                correct['iso_perm'] += 1

            iso_perm_both_vars = nx.is_isomorphic(G1, G2,
                node_match=categorical_node_match_permissive_both_vars("label"),
                edge_match=categorical_node_match_strict("label"))
            if iso_perm_both_vars:
                correct['iso_perm_both_vars'] += 1

            iso_perm_at_least_one_var = nx.is_isomorphic(G1, G2,
                node_match=categorical_node_match_permissive_at_least_one_var("label"),
                edge_match=categorical_node_match_strict("label"))
            if iso_perm_at_least_one_var:
                correct['iso_perm_at_least_one_var'] += 1

    correct_df = pd.DataFrame(correct.values(), columns=['accuracy'], index=correct.keys())
    correct_df['normalized'] = correct_df['accuracy'] / total

    display = """{}
Predicted queries evaluation:
{}
""".format('-'*30, correct_df.to_markdown())
    return display, correct_df

def patterns(df):
    """
    Calculates metrics about pattern prediction
    """
    # confusion matrix
    tots = df['patterns'].value_counts()
    idx = sorted(tots.index)
    cols = sorted(list(df['predicted_pattern'].value_counts().index)) + ['nan', 'tot']
    tot = df['patterns'].shape[0]
    cm = pd.DataFrame(np.zeros((len(idx), len(cols))),
        index = idx,
        columns=cols)
    cm.index.name = 'gold\\'
    for i, row in df[['patterns', 'predicted_pattern']].iterrows():
        y = row['predicted_pattern']
        if type(y) != str and (y is None or np.isnan(y)):
            y = 'nan'
        cm.loc[row['patterns'], y] += 1
    cm['tot'] = tots

    accuracy = np.diag(cm[idx]).sum()/tot

    recall = (np.diag(cm[idx]) / tots.sort_index()).mean()

    tot_pred = cm[idx].apply(sum, axis=0)

    precision = (np.diag(cm[idx]) / tot_pred.sort_index()).mean()

    f1 = 2 * precision * recall / (precision + recall)

    normalized_cm = cm[sorted(set(cols) - {'tot', 'nan'}) + ['nan']].div(cm['tot'], axis=0)

    display = """{}
Pattern prediction performance:
Accuracy: {}
Precision: {}
Recall: {}
F1: {}
Normalized Confusion matrix:
{}
Confusion matrix:
{}
""".format('-'*30, accuracy, precision, recall, f1, normalized_cm.to_markdown(), cm.to_markdown())
    return display, accuracy, normalized_cm, cm

def question_types(df):
    tots = df['correct_question_type'].value_counts()
    tot = df.shape[0]
    idx = sorted(tots.index)
    cols = sorted(list(df['predicted_question_type'].value_counts().index)) + ['nan', 'tot']
    cm = pd.DataFrame(np.zeros((len(idx), len(cols))),
        index = idx,
        columns=cols)
    cm.index.name = 'gold'
    for i, row in df[['correct_question_type', 'predicted_question_type']].iterrows():
        y = row['predicted_question_type']
        if (not isinstance(y, str)) and (y is None or np.isnan(y)):
            y = 'nan'
        cm.loc[row['correct_question_type'], y] += 1

    cm['tot'] = tots

    accuracy = np.diag(cm[idx]).sum()/tot

    recall = (np.diag(cm[idx]) / tots.sort_index()).mean()

    tot_pred = cm[idx].apply(sum, axis=0)

    precision = (np.diag(cm[idx]) / tot_pred.sort_index()).mean()

    f1 = 2 * precision * recall / (precision + recall)

    normalized_cm = cm[sorted(set(cols) - {'tot', 'nan'}) + ['nan']].div(cm['tot'], axis=0)

    display = """{}
Question type prediction performance:
Accuracy: {}
Precision: {}
Recall: {}
F1: {}
Normalized Confusion matrix:
{}
Confusion matrix:
{}
""".format('-'*30, accuracy, precision, recall, f1, normalized_cm.to_markdown(), cm.to_markdown())
    return display, accuracy, normalized_cm, cm

def to_jsonl(df, outpath):
    with open(outpath, 'w') as fd:
        for i, row in df.iterrows():
            line = json.dumps({"index": i, "query": row["predicted_query"]}) + "\n"
            fd.write(line)

def entities(df):
    precisions = []
    recalls = []
    f1s = []
    tot = df.shape[0]
    def angularsCheck(entity):
        if entity[0] != '<':
            entity = '<' + entity
        if entity[-1] != '>':
            entity = entity + '>'
        return entity
    def singleMetrics(el_gold, el_pred):
        es_gold = set(angularsCheck(e) for e in el_gold) if el_gold is not None else set()
        es_pred = set(angularsCheck(e) for e in el_pred) if el_pred is not None else set()

        intersection = es_gold & es_pred

        precision = len(intersection) / (len(es_pred) + sys.float_info.min)
        recall = len(intersection) / (len(es_gold) + sys.float_info.min)
        f1 = 2 * precision * recall / (precision + recall + sys.float_info.min)

        return precision, recall, f1
    for i, row in df[['correct_entities', 'predicted_entities']].iterrows():
        p, r, f1 = singleMetrics(row['correct_entities'], row['predicted_entities'])
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    resdf = pd.DataFrame({
        'avgPrecision': ['{:.3f} +- {:.3f}'.format(
                sum(precisions) / tot,
                statistics.stdev(precisions)
            )],
        'avgRecall': ['{:.3f} +- {:.3f}'.format(
                sum(recalls) / tot,
                statistics.stdev(recalls)
            )],
        'avgF1': ['{:.3f} +- {:.3f}'.format(
                sum(f1s) / tot,
                statistics.stdev(f1s)
            )]
    }, index = ['mean +- std']).transpose()

    display = """{}
Entities linking performance:
{}
""".format('-'*30, resdf.to_markdown())
    return display, resdf

def predicates(df):
    precisions = []
    recalls = []
    f1s = []
    tot = df.shape[0]
    def angularsCheck(entity):
        if entity[0] != '<':
            entity = '<' + entity
        if entity[-1] != '>':
            entity = entity + '>'
        return entity
    def singleMetrics(el_gold, el_pred):
        es_gold = set(angularsCheck(e) for e in el_gold) if el_gold is not None else set()
        es_pred = set(angularsCheck(e) for e in el_pred) if el_pred is not None else set()

        intersection = es_gold & es_pred

        precision = len(intersection) / (len(es_pred) + sys.float_info.min)
        recall = len(intersection) / (len(es_gold) + sys.float_info.min)
        f1 = 2 * precision * recall / (precision + recall + sys.float_info.min)

        return precision, recall, f1
    for i, row in df[['correct_predicates', 'predicted_predicates_from_graph']].iterrows():
        p, r, f1 = singleMetrics(row['correct_predicates'], row['predicted_predicates_from_graph'])
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    resdf = pd.DataFrame({
        'avgPrecision': ['{:.3f} +- {:.3f}'.format(
                sum(precisions) / tot,
                statistics.stdev(precisions)
            )],
        'avgRecall': ['{:.3f} +- {:.3f}'.format(
                sum(recalls) / tot,
                statistics.stdev(recalls)
            )],
        'avgF1': ['{:.3f} +- {:.3f}'.format(
                sum(f1s) / tot,
                statistics.stdev(f1s)
            )]
    }, index = ['mean +- std']).transpose()

    display = """{}
Predicates choice performance:
{}
""".format('-'*30, resdf.to_markdown())
    return display, resdf

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
    df = pd.read_json(input, orient='records', lines=True)
    if action == 'eval':
        print(patterns(df)[0])
        print(question_types(df)[0])
        print(entities(df)[0])
        print(predicates(df)[0])
        print(isom(df)[0])
    elif action == 'prepare-queries':
        if output_file is None:
            raise Exception('output_file cannot be None for prepare-queries')
        print(f"Writing queries to {output_file} for apache jena...")
        to_jsonl(df.loc[df["predicted_query"].notna(), ["predicted_query"]], output_file)
    elif action == 'merge-graphs':
        if graphs_input_file is None:
            raise Exception('graphs_input_file cannot be None for merge-graphs')
        if output_file is None:
            raise Exception('output_file cannot be None for merge-graphs')
        input_graph_df = pd.read_json(graphs_input_file, orient='records', lines=True)
        input_graph_df.index = input_graph_df['index']
        input_graph_df.drop(columns='index', inplace=True)
        input_graph_df.rename(columns={"graph": "predicted_graph"}, inplace=True)
        df = df.join(input_graph_df)

        df.to_json(output_file, orient='records', lines=True)


if __name__ == '__main__':
    main()