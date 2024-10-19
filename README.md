# MalDetect: A Deep Learning application for Malaria Detection

This repository contains the application that develops a deep-learning model to predict malaria from blood cell images. The model is trained on a dataset of blood cell images infected with malaria and uninfected cells. The model is then wrapped in a Flask web application. The project is containerized using Docker and the Docker image is deployed on the Cloud Run service. The web application allows users to upload a picture of a cell and get a prediction of whether the cell is infected with malaria or not.

## Dataset

The project utilizes the malaria cell images dataset available on Kaggle and TensorFlow datasets. This dataset includes images of cells infected with malaria and uninfected cells. This repository's `data/` directory provides the dataset link and description.

## Project Structure

The project is organized as follows:

- `artifacts/`: This directory contains the serialized model as `malaria_model.keras` file.

- `assets/`: This directory contains screenshots of the web application for testing and cloud deployment, EDA and model training and tracking using Comet ML, and tensorboard logs for the model.

- `notebook/`: This directory contains the Jupyter notebooks for data preprocessing, data augmentation, model training, and performance evaluation.

  - `EDA_MALARIA_CELLS.ipynb`: This notebook contains the code for exploratory data analysis.
  - `MODEL_TRAINING(EfficientNet).ipynb`: This notebook contains the EfficientNet model training and evaluation code.
  - `MODEL_TRAINING_CometML(EfficientNet).ipynb`: This notebook contains the code for experiment tracking using Comet ML.

- `data/`: This directory contains the dataset link and description used for training the model.

- `src/`: This directory contains the source code for the model training and prediction pipeline along with different components and utilities.

  - `components/`: This directory contains the custom components for the model training and prediction pipeline.

    - `dataset.py`: This file contains the code for loading and splitting the dataset.
    - `augmentation.py`: This file contains the code for data augmentation applied to the dataset during training.
    - `model.py`: This file contains the code for the model architecture used for the training.
    - `optimizer.py`: This file contains the code for the optimizer tuning used in the model training pipeline.

  - `pipeline/`: This directory contains the model training and prediction pipeline code.

    - `train_pipeline.py`: This file contains the code for the model training pipeline.
    - `predict_pipeline.py`: This file contains the code for the prediction pipeline.

  - `config.py`: This file contains the model training and prediction pipeline configuration parameters.
  - `utils.py`: This file contains the utility functions used in the model training and prediction pipeline.
  - `logger.py`: This file contains the code for logging during training and prediction.
  - `exception.py`: This file contains the custom exceptions used in the project.

- `static/`: This directory contains the web application's CSS stylesheet and JavaScript files.

  - `css/`: This directory contains the custom CSS stylesheet for the web application.
  - `js/`: This directory contains the image upload JavaScript file for the web application.

- `templates/`: This directory contains the HTML templates for the web application. The templates are:

  - `index.html`: This file contains the code for the web application's home page. It includes a brief description of the project and an image upload placeholder.
  - `layout.html`: This file contains the code for the web application layout. It is used as a base template for the other templates.
  - `result.html`: This file contains the code for the web application's prediction page, which displays the prediction results.

- `test/`: This directory contains the pytest tests for the Flask web application.

  - `test_app.py`: This file contains the tests for the Flask web application using the `pytest`. It includes tests for the home page and the prediction page.

- `test_images/`: This directory contains the test images for testing the prediction task.
- `app.py`: This file contains the code for the Flask web application. It contains the routes for the home page and the prediction page.

- `setup.py`: This file contains the project's setup configuration. It can be installed as a package using the `pip` package manager.
- `.env.example`: This file contains the example environment variables for the Flask web application. It is used as a template for the actual `.env` file.
- `app.yaml`: This file contains the configuration for deploying the Flask app on Google Cloud Platform (GCP).
- `Dockerfile`: This file contains the instructions for building the Docker image for the project.
- `.dockerignore`: This file contains the files to be ignored by Docker.
- `.gcloudignore`: This file contains the files that Google Cloud will ignore.
- `.gitignore`: This file contains the files to be ignored by Git.
- `requirements.txt`: This file contains the list of Python dependencies for the project. It can install the dependencies using the `pip` package manager.
- `requirements-test.txt`: This file contains the list of Python dependencies for testing the project. It can install the dependencies using the `pip` package manager.
- `LICENSE`: This file contains the license information for the project.
- `README.md`: This file provides an overview of the project and its structure.

## Getting Started (Super Quick Start)

To get started with the project, without too much hassle, follow these steps (not ordered necessarily):

1. Visit the repository: `git clone https://github.com/sitamgithub-MSIT/MalDetect.git`
2. Under the notebook folder, you will find the `EDA_MALARIA_CELLS.ipynb` and `MODEL_TRAINING(EfficientNet).ipynb` and `MODEL_TRAINING_CometML(EfficientNet).ipynb` notebooks. These notebooks contain the code for data preprocessing, data augmentation, model training, and performance evaluation.
3. Run those notebooks to perform EDA and model training and evaluation. Google Colab can be used to run the notebooks, and a T4 GPU configuration is sufficient.
4. Just upload the notebooks to Google Colab and run them in the Colab environment.
5. Install the required dependencies using the `requirements.txt` file in the colab environment.
6. Happy notebooking!

**Note**: You need a Comet ML account and API key to run the Comet ML notebook. The Comet ML account can be created [here](https://www.comet.ml/), and the API key can be found in the Comet ML account settings.

## Dependencies

The project requires the following dependencies to run:

- Python 3.10.8
- NumPy
- Matplotlib
- TensorFlow
- Keras
- Comet ML
- Flask
  ...and more.

Please refer to the `requirements.txt` file for the complete list of dependencies. The project also refers to the `requirements-test.txt` file for testing.

## Installation and Environment Setup

To install the required dependencies and set up the environment, follow these steps:

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/MalDetect.git`
2. Change the directory: `cd MalDetect`
3. Create a virtual environment: `python -m venv dlproj`
4. Activate the virtual environment:
   - For Windows: `dlproj\Scripts\activate`
   - For Linux/Mac: `source dlproj/bin/activate`
5. Install the required dependencies: `pip install -r requirements.txt`
6. Run the Flask app: `python app.py`

Now, you can just open up your local host and you should see the web application running. If you would like more information, please refer to the Flask documentation [here](https://flask.palletsprojects.com/en/2.0.x/quickstart/#debug-mode).

## Deployment

**Containerization**: The project is containerized using Docker. The Docker image is built using the `Dockerfile` in the project's root directory. That configuration file contains the instructions for building the Docker image while deploying the service in the cloud. Currently, only Google Cloud Platform (GCP) is tested for docker image deployment. Other cloud platforms are not sure about the deployment. Also, one can run the docker image locally with their preferences and own configurations. For more follow the instructions in the blog post [here](https://dev.to/pavanbelagatti/a-step-by-step-guide-to-containerizing-and-deploying-machine-learning-models-with-docker-21al).

**Google Cloud Platform (GCP)**: The project is deployed on the Google Cloud Platform (GCP) using the Cloud Run service. The Docker image is deployed on the Cloud Run service. To deploy the service to Google Cloud, a very brief overview of some of the steps is provided below:

- Sign up for a Google Cloud account.
- Set up a project and enable the necessary APIs (Create a new project in the Google Cloud Console.
  Enable the required APIs, such as Cloud Run and Artifact Registry, through the Console.)
- Deploy the Docker image to Google Cloud Run. (Build and push your Docker image to the Google Artifact Registry or another container registry. Deploy the image to Cloud Run by specifying the necessary configurations.)
- Access the service using the provided URL. (Once deployed, a URL is provided to access the service. Use the URL to access the service.)

For detailed instructions and code examples, please look at the blog post [here](https://lesliemwubbel.com/setting-up-a-flask-app-and-deploying-it-via-google-cloud/). I think the blog post should be enough to get you started with deploying the service on GCP. Also, please look at the screenshots in the assets folder for the deployment results of this project.

## Testing

To test the deployed service locally, follow these steps:

1. cd into the `test/` directory: `cd test/`
2. Assuming you have your conda environment activated, install the dependencies for testing: `pip install -r requirements-test.txt`
3. Run the `test_app.py` file to test the Flask app.
4. Execute the command: `pytest test_app.py`
5. Verify the response and check for any errors or issues.
6. Optionally, please look at the screenshots in the assets folder for test results.

## Model Training and Evaluation

The model training and evaluation process is documented in the Jupyter notebooks in the `notebook/` directory. These notebooks provide step-by-step instructions on data augmentation, data preprocessing, model selection, and performance evaluation. Then these notebooks are converted into Python scripts and saved in the `src/` directory. The model training pipeline is implemented in the `train_pipeline.py` file. The model prediction pipeline is implemented in the `predict_pipeline.py` file.

To train the model, follow these steps:

1. Run the `train_pipeline.py` file: `python -m src.pipeline.train_pipeline`
2. The model will be trained on the dataset and the serialized model will be saved in the `artifacts/` directory.
3. Comet ML and TensorBoard will log the model evaluation metrics.
4. The model can be used for prediction by running the flask app through `app.py` which uses the saved model and uses `predict_pipeline.py` for prediction.

**Note**: The model training and evaluation process can be customized by changing the hyperparameters and configurations in the config.py file. Comet ML can help track this process in more detail.

## Results

The trained model, with set hyperparameters, achieved an accuracy of 94.10% on the evaluation. Various runs were performed using Comet ML for experiment tracking. The best model was selected based on the validation accuracy and loss. The model is saved as a keras file provided in the repository itself. Further, that was applied in the prediction task and connected with the Flask app.

**Note**: For model runs, tensorboard and comet ml were used to visualize the training, validation loss, and accuracy in more detail. The screenshots of the tensorboard logs and comet ml logs are provided in the `assets/` folder.

## Conclusion

In this project, we successfully developed a deep-learning model that can predict malaria from blood cell images. Then we wrapped the model in a Flask web application and containerized it using Docker. Finally, we deployed it on the Google Cloud Platform (GCP) using the Cloud Run service.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or suggestions regarding the project, feel free to reach out to me on my [GitHub profile](https://github.com/sitamgithub-MSIT).

Happy coding!
