from flask import Flask
from api.controller import prediction_app
def create_app() -> Flask:

    flask_app = Flask('ml_api')

    from api.controller import prediction_app
    flask_app.register_blueprint(prediction_app)

    return flask_app