# Testing imports
import pytest
from app import app


@pytest.fixture
def client():
    """
    Creates a test client for the Flask app with testing mode enabled.

    Returns:
        FlaskClient: The test client for the Flask app.
    """

    # Setting the app to testing mode
    app.config["TESTING"] = True
    yield app.test_client()


def test_main_page(client):
    """
    Test the main page of the Malaria Detection App.

    Args:
        client: The test client for making HTTP requests.

    Returns:
        None

    Raises:
        AssertionError: If the response status code is not 200 or if the expected
                       message is not found in the response data.
    """

    # Sending a GET request to the main page
    response = client.get("/")

    # Checking if the response status code is 200 and if the expected message is in the response data
    assert response.status_code == 200
    assert b"Malaria Cell Detection" in response.data


def test_predict_image_file_success(client):
    """
    Test case for successful prediction.

    Args:
        client: The client object for making HTTP requests.

    Returns:
        None
    """

    # Sending a POST request to the prediction route with the test image
    with open("test_images\parasite0.png", "rb") as image_file:
        response = client.post("/prediction", data={"file": image_file})

    # Checking if the response status code is 200 and if the expected message is in the response data
    assert response.status_code == 200
    assert b"Prediction" in response.data
