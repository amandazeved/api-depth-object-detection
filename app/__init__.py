from flask import Flask

def create_app():
    app = Flask(__name__)

    # importando e registrando blueprints
    from app.main import main_bp
    app.register_blueprint(main_bp)

    return app