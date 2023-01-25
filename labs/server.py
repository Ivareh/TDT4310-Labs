# This is the main server file for the lab.
# You should not need to change anything here.
import flask
from flask_cors import CORS, cross_origin
from lab_runner import LabRunner

server = flask.Flask(__name__)
cors = CORS(server)

FRONTEND_HOST = "localhost"
FRONTEND_PORT = 3000
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080

server.config['CORS_HEADERS'] = 'Content-Type'
server.config['CORS_ORIGINS'] = '*'

lab = LabRunner()

def make_response(error: bool, message: str) -> flask.Response:
    return flask.jsonify({ "error": error, "message": message })

@server.route('/')
def health_check() -> str:
    return f"Server is running on {SERVER_HOST}:{SERVER_PORT}"

# allow cross-origin requests for the predictions endpoint
@server.route('/predictions', methods=['POST'])
@cross_origin()
def predictions() -> flask.Response:
    text = flask.request.json.get('text')
    preds = lab.predict(text)
    return flask.jsonify(preds)

@server.route('/train', methods=['POST'])
@cross_origin()
def train() -> flask.Response:
    lab_number = flask.request.json.get('lab')
    if lab_number is None:
        return make_response(True, "No lab number provided")

    lab.train(lab_number)
    return make_response(False, f"Lab {lab_number + 1} is now trained")

@server.route('/lab', methods=['POST'])
@cross_origin()
def set_lab() -> flask.Response:
    lab_number = flask.request.json.get('lab')
    if lab_number is None:
        return make_response(True, "No lab number provided")
    lab.set_active(lab_number)
    return make_response(False, f"Lab {lab_number + 1} is now active")

@server.route('/status', methods=['GET'])
@cross_origin()
def status() -> flask.Response:
    status_info = {
        "text": lab.current_text,
        "trained": [bool(lab) for lab in lab.initialized_labs],
    }
    print("Status info: ", status_info)
    return flask.jsonify(status_info)

server.run(host=SERVER_HOST, port=SERVER_PORT, debug=True, threaded=True)
