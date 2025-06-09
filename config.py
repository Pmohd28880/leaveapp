import logging
from logging.handlers import RotatingFileHandler


def configure_logging(app):
    # Remove all handlers
    for handler in app.logger.handlers[:]:
        app.logger.removeHandler(handler)

    # Set up file logging
    file_handler = RotatingFileHandler(
        'leave_system.log',
        maxBytes=1024 * 1024,
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)

    # Set log level
    app.logger.setLevel(logging.INFO)
