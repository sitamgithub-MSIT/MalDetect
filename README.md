# MedHelp: A Machine Learning Model for Predicting Medical Costs

This repository contains the ML ZoomCamp Capstone Project 1, which focuses on developing a machine learning model to predict medical costs for individuals based on various factors such as age, BMI, smoking status, and region. The project aims to provide accurate predictions for insurance companies to estimate premiums and individuals to make informed healthcare decisions.

## Dataset

The project utilizes the Medical Cost Personal Datasets available on Kaggle. This dataset includes information about individuals' medical costs, as well as their demographic and lifestyle attributes. The dataset is provided in a CSV format and can be found in the `data/` directory under the notebook folder of this repository as `insurance.csv`.

## Project Structure

The project is organized as follows:

- `artifacts/`: This directory contains the serialized model and the data preprocessing objects as pickle files named `training_model.pkl`, and `preprocessor.pkl` respectively.

- `assets/`: This directory contains the screenshots of the web application for testing and cloud deployment.

- `notebooks/`: This directory contains the Jupyter notebooks for data preprocessing, feature engineering, model selection, and performance evaluation.

  - `1. EDA INSURANCE.ipynb`: This notebook contains the code for exploratory data analysis.
  - `2. MODEL TRAINING.ipynb`: This notebook contains the code for feature engineering and model training.
  - `data/`: This directory contains the dataset used for training the model.

- `src/`: This directory contains the source code for utility functions, helper classes, and the project's components and pipeline definition.

  - `components/`: This directory contains the code for the project's components. Each component is defined in a separate file. The components are:
    - `data_ingestion.py`: This file contains the code for data ingestion and train-test split.
    - `data_transformation.py`: This file contains the code for feature engineering and data transformation.
    - `model_training.py`: This file contains the code for model training, hyperparameter tuning, and performance evaluation.
  - `pipeline/`: This directory contains the code for the project's pipeline definition. Each pipeline is defined in a separate file. The pipelines are:
    - `train_pipeline.py`: This file should be used for pipeline training. Currently, it is empty and should be filled with the appropriate code.
    - `predict_pipeline.py`: This file contains the code for the prediction pipeline. It loads the serialized model and the data preprocessing objects and uses them to make predictions. Also, it contains custom data class for the prediction request.
  - `exception.py`: This file contains the custom exceptions for the project.
  - `logger.py`: This file contains the code for logging.
  - `utils.py`: This file contains the utility functions for the project.

- `templates/`: This directory contains the HTML templates for the web application. The templates are:

  - `home.html`: This file contains the code for the home page of the web application. It contains a brief description of the project and a link to the prediction page.
  - `prediction.html`: This file contains the code for the prediction page of the web application. Contains a form for entering the input data. Also, it contains the code for displaying the prediction results.

- `app.py`: This file contains the code for the Flask web application. It contains the routes for the home page and the prediction page.
- `test_app.py`: This file contains the tests for the Flask web application using the `pytest`. It contains tests for the home page and the prediction page.
- `Dockerfile`: This file contains the instructions for building the Docker image for the project.
- `requirements.txt`: This file contains the list of Python dependencies for the project. It can be used to install the dependencies using the `pip` package manager.
- `setup.py`: This file contains the instructions for installing the project as a Python package.
- `README.md`: This file provides an overview of the project and its structure.

## Getting Started (Super Quick Start)

To get started with the project, without too much hassle, follow these steps:

1. Visit the repository: `git clone https://github.com/sitamgithub-MSIT/capstone-project1.git`
2. Under notebooks folder, you will find the `1. EDA INSURANCE.ipynb` and `2. MODEL TRAINING.ipynb` notebooks. These notebooks contain the code for data preprocessing, feature engineering, model selection, hyperparameter tuning, and performance evaluation.
3. Run those notebooks to perform EDA and model training and evaluation. Google Colab can be used to run the notebooks. CPU configuration is sufficient to run the notebooks.
4. Just upload the notebooks to Google Colab and run them in the Colab environment.
5. Make sure to upload the dataset to the Colab environment. Also, don't forget to change the path of the dataset in the notebooks.
6. Install the required dependencies using the `requirements.txt` file in colab environment.
7. Happy notebooking!

## Dependencies

The project requires the following dependencies to run:

- Python 3.8
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
  ...and more.

Please refer to the `requirements.txt` file for the complete list of dependencies. And for testing, the project refers to the `requirements-test.txt` file.

## Installation and Environment Setup

To install the required dependencies and set up the environment, follow these steps:

1. Clone the repository: `git clone https://github.com/sitamgithub-MSIT/capstone-project1.git`
2. Change to the project directory: `cd MLCapstone1`
3. Create a virtual environment: `python -m venv env`
4. Activate the virtual environment:
   - For Windows: `.\env\Scripts\activate`
   - For macOS/Linux: `source env/bin/activate`
5. Install the required dependencies: `pip install -r requirements.txt`

## Testing

To test the deployed service locally, follow these steps:

1. cd into the project directory.
2. Assuming you have your conda environment activated, install the dependencies for testing: `pip install -r requirements-test.txt`
3. Run the `test_app.py` file to test the Flask app.
4. Execute the command: `pytest test_app.py`
5. Verify the response and check for any errors or issues.
6. Optionally, refer to the screenshots in the assets folder for test results.

## Model Training and Evaluation

The model training and evaluation process is documented in the Jupyter notebooks in the `notebooks/` directory. These notebooks provide step-by-step instructions on data preprocessing, feature engineering, model selection, hyperparameter tuning, and performance evaluation. Then those notebooks are converted to the python scripts and placed in the `src/components/` directory. The `train_pipeline.py` file is used to train the pipeline. The `predict_pipeline.py` file is used to make predictions using the trained model.

## Results

The trained model, with hyperparameter tuning, achieved an accuracy of 89% on the test set. Various models were trained and evaluated, and the best model was selected based on the performance metrics. Various performance metrics were used to evaluate the models, such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R2). The best model was selected based on the evaluation of metrics. Model was then serialized and saved as a pickle file. The data preprocessing object was also serialized and saved as pickle files. The serialized model and the data preprocessing object are available in the `artifacts/` directory. Further that was applied in prediction pipeline conecting with Flask app.

## Conclusion

In this project, we successfully developed a machine learning model that can predict medical costs for individuals based on various factors such as age, BMI, smoking status, and region. Then we wrapped the model in a Flask web application and containerized it using Docker. Finally, we deployed it on the Google Cloud Platform (GCP) using the Cloud Run service.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions or suggestions regarding the project, feel free to reach out to me on my [GitHub profile](https://github.com/sitamgithub-MSIT).

Happy coding!
