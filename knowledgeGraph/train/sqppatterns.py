#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[2]:


import pandas as pd
import re
import json
import click
import os
import networkx as nx
import sys


def get_pattern_from_graph(graph, patterns):
    # convert into graph
    graph = nx.from_dict_of_lists(graph, create_using=nx.DiGraph)

    # find pattern
    for key, pattern in patterns.items():
        if nx.is_isomorphic(graph, pattern):
            return key

    # pattern not found
    return "p_notFound"

def get_patterns():
    p0 = {'A': []}
    p1 = {'A': ['B']}
    p2 = {'A': ['B'],
          'B': ['C']}
    p3 = {'A': ['B'],
          'C': ['B']}
    p4 = {'A': ['B', 'C']}
    p5 = {'A': ['B', 'C', 'D']}
    p6 = {'A': ['B', 'C'],
          'C': ['D']}
    p7 = {'A': ['B'],
          'B': ['C'],
          'C': ['D']}
    p8 = {'A': ['B'],
          'B': ['C'],
          'D': ['C']}
    p9 = {'A': ['B'],
          'B': ['C'],
          'D': ['B']}
    p10 = {'A': ['B'],
           'B': ['C', 'D']}
    p11 = {'A': ['B'],
           'C': ['B'],
           'D': ['B']}
    # convert patterns into graphs
    g0 = nx.from_dict_of_lists(p0, create_using=nx.DiGraph)
    g1 = nx.from_dict_of_lists(p1, create_using=nx.DiGraph)
    g2 = nx.from_dict_of_lists(p2, create_using=nx.DiGraph)
    g3 = nx.from_dict_of_lists(p3, create_using=nx.DiGraph)
    g4 = nx.from_dict_of_lists(p4, create_using=nx.DiGraph)
    g5 = nx.from_dict_of_lists(p5, create_using=nx.DiGraph)
    g6 = nx.from_dict_of_lists(p6, create_using=nx.DiGraph)
    g7 = nx.from_dict_of_lists(p7, create_using=nx.DiGraph)
    g8 = nx.from_dict_of_lists(p8, create_using=nx.DiGraph)
    g9 = nx.from_dict_of_lists(p9, create_using=nx.DiGraph)
    g10 = nx.from_dict_of_lists(p10, create_using=nx.DiGraph)
    g11 = nx.from_dict_of_lists(p11, create_using=nx.DiGraph)

    # create dictionary containing all patterns
    patterns = {
        'p0': g0,
        'p1': g1,
        'p2': g2,
        'p3': g3,
        'p4': g4,
        'p5': g5,
        'p6': g6,
        'p7': g7,
        'p8': g8,
        'p9': g9,
        'p10': g10,
        'p11': g11
    }

    return patterns


@click.command()
@click.option('--action', type=click.Choice(['prepare-query', 'get-pattern', 'prepare-evaluation'], case_sensitive=False), help='Main action')
@click.option('-i', '--input-file', help='Input file containing data in json format. Needed by both actions.')
@click.option('-o', '--output-file',
              help='Output file created by the chosen action.'
              'For "prepare-query" it contains one query per line so that apache jena is able to process them.'
              'For "get-pattern" it is a csv containing all fields from the input data together with'
              'the extracted patter')
@click.option('-g', '--input-file-graph', help='Input file containing query graphs created by apache jena')
def go(action, input_file, output_file, input_file_graph):
    """
    # Prepare query file for apache jena: starting from an input json file like the ones from lc-quad generate. A file containing one query per line ready to be processed by the apache jena app

    ./sqppatterns.py --action=prepare-query --input-file /path/to/input-file.json --output-file=/path/to/output-file.csv

    # Extract SQP patterns from the query graph file obtained by the apache jena app. input-file-graph.json is the output of apache jena

    ./sqppatterns.py --action=get-pattern --input-file /path/to/input-file.json --input-file-graph /path/to/input-file-graph.json --output-file=/path/to/output-file.csv
    """
    if not input_file:
        print("ERROR: '--input-file' must be specified.")
        sys.exit(1)

    if not output_file:
        print("ERROR: '--output-file' must be specified.")
        sys.exit(1)

    if action == 'prepare-query':
        data = pd.read_json(input_file)

        print(f"Writing queries to {output_file} for apache jena...")
        data[["sparql_query"]].to_csv(output_file, header=None, index=None)

    elif action == 'get-pattern':

        if not input_file_graph:
            print("ERROR: '--input-file-graph' must be specified.")
            sys.exit(1)

        data = pd.read_json(input_file)

        graph_df = pd.read_table(
            input_file_graph, header=None, names=["graph"])
        graph_df["dicts"] = graph_df["graph"].apply(lambda x: json.loads(x))

        # extract patterns
        patterns = get_patterns()
        graph_df["patterns"] = graph_df["dicts"].apply(lambda x: get_pattern_from_graph(x, patterns))

        print("Value counts:")
        print(graph_df["patterns"].value_counts())

        # save patterns
        data["patterns"] = graph_df["patterns"]

        print(f"Writing data with patterns to {output_file}...")
        data.to_csv(output_file)

    elif action == 'prepare-evaluation':

        if not input_file_graph:
            print("ERROR: '--input-file-graph' must be specified.")
            sys.exit(1)

        data = pd.read_json(input_file)

        graph_df = pd.read_table(
            input_file_graph, header=None, names=["graph"])
        graph_df["dicts"] = graph_df["graph"].apply(lambda x: json.loads(x))

        # import pdb
        # pdb.set_trace()

        try:
            testd = graph_df["dicts"].iloc[0]
            testd_inner = testd[next(iter(testd.keys()))]
            testd_inner_inner = testd_inner[next(iter(testd_inner.keys()))]
        except AttributeError as e:
            print("""ERROR: --input-file-graph json lines have too few levels.
            Probably you passed a `dict_to_lists` like file.
            Instead a `dict_to_dicts` like file is required!
            """, file=sys.stderr)
            sys.exit(1)

        # extract patterns
        patterns = get_patterns()
        graph_df["patterns"] = graph_df["dicts"].apply(lambda x: get_pattern_from_graph(x, patterns))

        print("Value counts:")
        print(graph_df["patterns"].value_counts())

        # save patterns
        data["patterns"] = graph_df["patterns"]

        # save graphs as strings
        data["graph"] = graph_df["graph"]

        print(f"Writing data with patterns to {output_file}...")
        data.to_csv(output_file)

    else:
        print("ERROR '--action' must be specified.")
        sys.exit(1)

go()
