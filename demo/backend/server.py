import os
import sys
import networkx as nx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
from flask import Flask
from flask_cors import CORS

from utils import load_models, to_dict_of_dicts
from flask import request, abort, jsonify


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
        except:
            abort(500, description="Pattern could not be classified.")

        # extract and link entities
        entities, texts = entity_extractor.extract(question)
        print('Extracted entities:', entities)
        print('Extracted texts:', texts)
        if not entities:
            abort(500, description="Could not identify any entity.")

        entities_copy = entities.copy()

        # query graph construction
        try:
            Q = query_graph_builder.build(question, entities, patterns[0])
        except:
            abort(500, description="Could not construct the query graph.")

        # build SPARQL query and retrieve answers
        try:
            SPARQL_query = query_generator.generate(question, Q)
        except:
            abort(500, description="Could not generate the SPARQL query.")
        
        try:
            answers_df = query_generator.ask(SPARQL_query)
        except:
            abort(500, description="Could not query DBPedia.")
        #answers_df = query_generator.generate_and_ask(question, Q)

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
    # free-text question answering
    return

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        pattern_classifier, entity_extractor, query_graph_builder, query_generator = load_models()
    app.run(debug=True)