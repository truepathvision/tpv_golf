import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

if __name__ == "__main__":
    app, window = create_app()
    window.show()
    sys.exit(app.exec())
