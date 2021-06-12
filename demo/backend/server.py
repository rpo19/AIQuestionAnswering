import os
import sys
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
from flask import Flask
from flask_cors import CORS

from utils import load_models_kgqa, load_models_ftqa, to_dict_of_dicts
from flask import request, abort, jsonify
import traceback

import logging
logging.getLogger().setLevel(logging.DEBUG)


app = Flask(__name__)
CORS(app)

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error=str(e)), 500

@app.errorhandler(400)
def bad_request(e):
    return jsonify(error=str(e)), 400

@app.route("/api/kgqa", methods=['GET'])
def ask_kgqa():
    question = request.args.get('q')
    if question:
        # classify question with pattern
        try:
            patterns, raw_probs = pattern_classifier.transform(question)
            print('Pattern:', patterns[0])
        except Exception as e:
            traceback.print_exc()
            print(e)
            abort(500, description="Pattern could not be classified.")

        # extract and link entities
        entities, texts = entity_extractor.extract(question)
        print('Extracted entities:', entities)
        print('Extracted texts:', texts)
        if not entities:
            print("Could not identify any entity.")
            abort(500, description="Could not identify any entity.")

        entities_copy = entities.copy()

        # query graph construction
        try:
            Q = query_graph_builder.build(question, entities, texts, patterns[0])
        except Exception as e:
            traceback.print_exc()
            print(e)
            abort(500, description="Could not construct the query graph.")

        # build SPARQL query and retrieve answers
        try:
            SPARQL_query, constraints = query_generator.generate(question, Q)
        except Exception as e:
            traceback.print_exc()
            print(e)
            abort(500, description="Could not generate the SPARQL query.")

        #try:
        answers_df = query_generator.ask(Q, entities, SPARQL_query, constraints)
        #except:
        #    abort(500, description="Could not query DBPedia.")

        answers = answers_df['Answers'].values.tolist()
        return {
            'pattern': patterns[0],
            'entities': entities_copy,
            'graph': to_dict_of_dicts(Q),
            'query': SPARQL_query,
            'answers': answers
        }

    abort(400, description="Query cannot be empty.")

@app.route("/api/ftqa", methods=['GET'])
def ask_ftqa():
    question = request.args.get('q')
    mode = request.args.get('mode')
    print('Mode:',mode)
    answers = []
    answerType = mode
    if question:
        if mode == 'NER and NEL':
            # extract and link entities
            entity, text = entity_extractor.extractMain(question)
            print('Extracted entities:', entity)
            print('Extracted texts:', text)
            # get answers from wikipedia
            answers = free_text_answerer.answerFromWiki(question, entity)
        elif mode == 'Span of text':
            span = request.args.get('span')
            if span:
                answers = free_text_answerer.answerFromSpan(question, span)
                answers = [answers]
        elif mode == 'Wikipedia search':
            answers = free_text_answerer.answerFromWiki(question)

    return {'answers': answers, 'answerType': answerType}

if __name__ == '__main__':
    if os.environ.get("PRODUCTION") is not None:
        pattern_classifier, entity_extractor, query_graph_builder, query_generator = load_models_kgqa()
        free_text_answerer = load_models_ftqa()
        app.run(host="0.0.0.0", port=5000)
    else:
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
            pattern_classifier, entity_extractor, query_graph_builder, query_generator = load_models_kgqa()
            free_text_answerer = load_models_ftqa()
        app.run(debug=True)
