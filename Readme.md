## Convolutional Neural Network for Flower Classification
This model contains flower speceies of over 100 classes for classification and also uses the Flask framework as its web front end to interact with this model.

### 1. Clone the repository
Clone the project by ruuning the below code from the terminal

`git clone https://github.com/princesegzy01/ConvNet-Flower-Classification.git`

this will clone the source code and datasets into a folder for the next stage

###2. Installing the packages needed
In order for you to not have issues installing all the packages with its version needed, All the required packages are inside the requirement.txt and you can bult install them with a single command.

`pip3 install -r requirement.txt`

### 3. Running from Command line
The model can be directly called from the python command line by running

`python3 classifier.py` 

This will automatically loop through all the images in the test folder and predict them by outputing the classes on the terminal

### 3. Predicting from Web Interface
You can also run the classifier from a web interface by following the steps below

1. run `python3 classifier.py` to train the classifier as this will aslo generates a model.h5 file which will be used by the flask frameworks
2. from the cli run `python3 run.py`. This will create a webserver with a port 5000
3. open any browser of your choice and log on to `http://127.0.0.1:5000` you should be welcomed with a simple page and upload button.
4. upload your flower image and click on the upload button, you will be redirected to the prediction page.
5. Thanks


contact for more information

princesegzy01@gmail.com
