# Importing required libs
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from model import preprocess_img, predict_result
from flask_cors import CORS

load_dotenv()

# Instantiating flask app
application = Flask(__name__)
app = application
CORS(app)

# Environment variables for the application configuration settings
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG")


# Home route for the app
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route for image file
@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """
    Endpoint for predicting the result of an image file.

    This function receives a POST request with an image file and preprocesses it.
    It then predicts the result using the `predict_result` function.
    The predicted result is rendered in the "result.html" template.

    Returns:
        str: The predicted result as a string.

    Raises:
        Exception: If the file cannot be processed.

    """

    # Try block for handling exceptions
    try:
        # Checking if the request method is POST
        if request.method == "POST":
            # Preprocessing the image file and predicting the result
            img = preprocess_img(request.files["file"].stream)
            pred = predict_result(img)
            return render_template("result.html", predictions=str(pred))

    except Exception:
        # Error message to be displayed if the file cannot be processed
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code for running the app
if __name__ == "__main__":
    app.run()
