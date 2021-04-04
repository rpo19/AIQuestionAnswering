import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../..')))
from flask import Flask
from flask_cors import CORS

from utils import load_models
from flask import request


app = Flask(__name__)
CORS(app)

@app.route("/api/kgqa", methods=['GET'])
def ask_kgqa():
    question = request.args.get('q')
    if question:
        # classify question with pattern
        patterns, raw_probs = pattern_classifier.transform(question)
        print('Pattern:', patterns[0])

        # extract and link entities
        entities, texts = entity_extractor.extract(question)
        entities_copy = entities.copy()
        print('Extracted entities:', entities)
        print('Extracted texts:', texts)

        # query graph construction
        Q = query_graph_builder.build(question, entities, patterns[0])

        # build SPARQL query and retrieve answers
        answers_df = query_generator.generate_and_ask(question, Q)

        answers = answers_df['Answers'].values.tolist()
        return { 'pattern': patterns[0], 'entities': entities_copy, 'answers': answers }
    
    return {'error': 'Question is empty!'}

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        pattern_classifier, entity_extractor, query_graph_builder, query_generator = load_models()
    app.run(debug=True)