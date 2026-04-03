from flask import Flask, request, jsonify
from flask_cors import CORS
from scout import handle_query

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400
    answer = handle_query(user_query)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(port=5000, debug=True)