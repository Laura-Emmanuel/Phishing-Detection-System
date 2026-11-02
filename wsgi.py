"""Cloud entry point for WSGI servers such as Gunicorn."""

from app.app import app


if __name__ == "__main__":
    app.run()
