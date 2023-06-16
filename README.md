# SkinSight-ML

Skin diseases are prevalent worldwide and can have a significant impact on a person's quality of life. Timely and accurate diagnosis of these conditions is essential for effective treatment and management. With advances in machine learning and computer vision, the development of reliable models for skin disease classification has become possible.

The main objective of this project is to create a classification model for skin diseases using the [DermNet](https://www.kaggle.com/datasets/shubhamgoel27/dermnet) dataset. The DermNet dataset provides a comprehensive collection of images representing various dermatological conditions. By leveraging this dataset, we aim to develop a model that can accurately classify skin diseases based on input images.

# About Dataset

The dataset used in this project includes a collection of images depicting 23 different types of skin diseases sourced from [DermNet](https://dermnetnz.org/) and have been collected [here](https://www.kaggle.com/datasets/shubhamgoel27/dermnet). The dataset contains a total of approximately 19.5k images, which have been divided into a training set of around 15k images and a separate test set.

# Model

[Old H5 Model Gdrive Link](https://drive.google.com/drive/folders/1WMwYZ6HjC6l8LIYJ3VYcoGsf9QxUnIQQ) (Old Model)

[Final H5 Model Gdrive Link](https://drive.google.com/drive/folders/1PwCnlkURBq8iMZZs-HOQWN8RdSg-o9WO) (Final Model)

# How To Use The Notebook
1. Open Google Colab. Go to https://colab.research.google.com/ and sign in with your Google account if necessary.
2. Create a New Notebook. Click on "File" in the top-left corner of the Colab interface and select "New Notebook" to create a new notebook.
3. You can clone this repo or just download the notebook.
4. Open the notebook in Google Colab.
5. Follow the instructions in the notebook and run the notebook.

# How to test Flask API using cmd
1. Download the Flask API folder (in master branch).
2. Download and copy model h5 to Flask directory.
3. Open cmd and navigate to Flask API directory.
4. Create a Python virtual environment by running this following command:
   
   `pip install virtualenv`
   
   then run this:
   
   `virtualenv skinsight`
   
    Note: skinsight is my virtual environment name

6. In file explorer, you can see virtual environment created successfully.
7. Activate virtual environment by running this following command:

   `skinsight\Scripts\activate.bat`

8. Left side we can see (skinsight) before file directory, that means virtual environment is active. Now virtual environment is activated, we will install required library for the project.
9. Install the required library by running this following command:

    `pip install flask`
    
    
    `pip install tensorflow`
    
    
    `pip install pillow`

    Note: make sure you have a compatible version of Python installed on your system as well.

10. Run the flask application

     `python main.py`

11. Server is running. Open url http://127.0.0.1:5000/ in browser.
   
