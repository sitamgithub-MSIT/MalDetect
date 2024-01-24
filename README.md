# MalDetect: A Deep Learning application for Malaria Detection

This repository contains the ML ZoomCamp Capstone Project 2, which focuses on developing a deep learning model to predict malaria from cell images. The project aims to provide accurate predictions for malaria detection using deep learning techniques.

## Dataset

The project utilizes the malaria cell images dataset available on Kaggle. This dataset includes images of cells infected with malaria and cells that are uninfected. The dataset link along with the dataset description is provided in the `data/` directory of this repository.

## Project Structure

The project is organized as follows:

- `artifacts/`: This directory contains the serialized model as `model.h5` file.

- `assets/`: This directory contains the screenshots of the web application for testing and cloud deployment. Also EDA and model training results are available in this folder.

- `notebook/`: This directory contains the Jupyter notebooks for data preprocessing, data augmentation, model training, and performance evaluation.

  - `1. EDA MALARIA.ipynb`: This notebook contains the code for exploratory data analysis.
  - `2. MODEL TRAINING(model_name).ipynb`: This notebook contains the code for model training and evaluation of various models.

- `data/`: This directory contains the dataset link and description used for training the model.

- `src/`: This directory contains the source code for the model training script of best performing model.

  - `train.py`: This file contains the code for model training and evaluation. It is the script for training the model based on the best performing model notebook.

- `static/`: This directory contains the CSS stylesheet and JavaScript files for the web application.

  - `css/`: This directory contains the custom CSS stylesheet for the web application.
  - `js/`: This directory contains the image upload JavaScript file for the web application.

- `templates/`: This directory contains the HTML templates for the web application. The templates are:

  - `index.html`: This file contains the code for the home page of the web application. It contains a brief description of the project and a image upload placeholder.
  - `layout.html`: This file contains the code for the layout of the web application. It is used as a base template for the other templates.
  - `result.html`: This file contains the code for the prediction page of the web application. It contains the code for displaying the prediction results.

- `test_images/`: This directory contains the test images for testing the prediction task.
- `model.py`: This file contains the code for various utility functions for the prediction task.
- `app.py`: This file contains the code for the Flask web application. It contains the routes for the home page and the prediction page.
- `test_app.py`: This file contains the tests for the Flask web application using the `pytest`. It contains tests for the home page and the prediction page.
- `Dockerfile`: This file contains the instructions for building the Docker image for the project.
- `requirements.txt`: This file contains the list of Python dependencies for the project. It can be used to install the dependencies using the `pip` package manager.
- `requirements-test.txt`: This file contains the list of Python dependencies for testing the project. It can be used to install the dependencies using the `pip` package manager.
- `README.md`: This file provides an overview of the project and its structure.

## Getting Started (Super Quick Start)

To get started with the project, without too much hassle, follow these steps (not ordered necessarily):

1. Visit the repository: `git clone https://github.com/sitamgithub-MSIT/capstone-project2.git`
2. Under notebook folder, you will find the `1. EDA MALARIA.ipynb` and `2. MODEL TRAINING(model_name).ipynb` notebooks. These notebooks contain the code for data preprocessing, data augmentation, model training, and performance evaluation.
3. Run those notebooks to perform EDA and model training and evaluation. Google Colab can be used to run the notebooks. T4 GPU configuration is sufficient to run the notebooks.
4. Just upload the notebooks to Google Colab and run them in the Colab environment.
5. Install the required dependencies using the `requirements.txt` file in colab environment.
6. Happy notebooking!

## Dependencies

The project requires the following dependencies to run:

- Python 3.9
- NumPy
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- Flask
  ...and more.

Please refer to the `requirements.txt` file for the complete list of dependencies. And for testing, the project refers to the `requirements-test.txt` file.

## Installation and Environment Setup

To install the required dependencies and set up the environment, follow these steps:

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/capstone-project2.git`
2. Create a virtual environment: `conda create -n dlproj python=3.9 -y`
3. Activate the virtual environment: `conda activate dlproj`
4. Install the required dependencies: `pip install -r requirements.txt`
5. Run the Flask app: `python app.py`

Now, open up your local host and you should see the web application running. For more information, refer to the Flask documentation [here](https://flask.palletsprojects.com/en/2.0.x/quickstart/#debug-mode).

## Deployment

**Containerization**: The project is containerized using Docker. The Docker image is built using the `Dockerfile` in the project's root directory. That configuration file contains the instructions for building the Docker image while deploying the service in the cloud. Currently, only Google Cloud Platform (GCP) is tested for docker image deployment. Other cloud platforms not sure about the deployment. Alos, one can run the docker image locally with their preferences and own configurations. For more follow the instructions in the blog post [here](https://dev.to/pavanbelagatti/a-step-by-step-guide-to-containerizing-and-deploying-machine-learning-models-with-docker-21al).

**Google Cloud Platform (GCP)**: The project is deployed on the Google Cloud Platform (GCP) using the Cloud Run service. The Docker image is deployed on the Cloud Run service. To deploy the service to Google Cloud, a very brief overview of some of the steps is provided below:

- Sign up for a Google Cloud account.
- Set up a project and enable the necessary APIs (Create a new project in the Google Cloud Console.
  Enable the required APIs, such as Cloud Run and Artifact Registry, through the Console.)
- Deploy the Docker image to Google Cloud Run. (Build and push your Docker image to Google Artifact Registry or another container registry. Deploy the image to Cloud Run by specifying the necessary configurations.)
- Access the service using the provided URL. (Once deployed, a URL is provided to access the service. Use the URL to access the service.)

For detailed instructions and code examples, please refer to the blog post [here](https://lesliemwubbel.com/setting-up-a-flask-app-and-deploying-it-via-google-cloud/). The blog post should be sufficient to get you started with deploying the service on GCP. Also, refer to the screenshots in the assets folder for deployment results of this project.

## Testing

To test the deployed service locally, follow these steps:

1. cd into the project directory.
2. Assuming you have your conda environment activated, install the dependencies for testing: `pip install -r requirements-test.txt`
3. Run the `test_app.py` file to test the Flask app.
4. Execute the command: `pytest test_app.py`
5. Verify the response and check for any errors or issues.
6. Optionally, refer to the screenshots in the assets folder for test results.

## Model Training and Evaluation

The model training and evaluation process is documented in the Jupyter notebooks in the `notebook/` directory. These notebooks provide step-by-step instructions on data augmentation, data perprocessing, model selection, and performance evaluation. Then the best performing model notebook is converted into a Python script and saved as `train.py`. The `model.py` file contains the code for various utility functions for the prediction task.

## Results

The trained model, with set hyperparameters, was able to achieve an accuracy of 96.5% on the evaluation set. Various models were trained and evaluated, and the best model was selected based on the performance metric. The best model was selected based on the evaluation of accuracy metric. The model saved as HDF5 file provided in the repository itself. Further that was applied in prediction task and connected with Flask app.

Note: For models, tensorboard was used to visualize the training and validation loss and accuracy in more detail. Though only for the best model, the tensorboard logs are available in the `assets/` directory.

## Conclusion

In this project, we successfully developed a deep learning model that can predict malaria from cell images. Then we wrapped the model in a Flask web application and containerized it using Docker. Finally, we deployed it on the Google Cloud Platform (GCP) using the Cloud Run service.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or suggestions regarding the project, feel free to reach out to me on my [GitHub profile](https://github.com/sitamgithub-MSIT).

Happy coding!
