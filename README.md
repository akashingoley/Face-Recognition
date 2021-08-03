# Face-Recognition
Using opencv to identify faces

It contains two files

### 1] Data Collection
In this file, you create a file of different faces using hardcascade_frontalface
The camera will start once you run the program and it can only be exited once you press 'q' key
Once face is recognised and the camera is closed you will be prompted to enter a name for a filename to recognise the face, that filename will be displayed later while classifying.

### 2] Classification
I have used KNN model to classify different faces.
Firstly it will identify different .npy files which will contain the face image data and append it to X data and a new y data will be created to represent different groups.
Both x and y data are concatenated to create training data for our model.
A separate dictionary is made with y data to represent the filename or the name of the person whose face we have to recognise.

#### Prediction
Once Classify file is run it will again start the camera and detect faces again by using Hardcascade_frontalface
It will create X_test points when the camera starts detecting face and it will run through our algorithm tp make a prediction.
A square box will be created with their name on top of the box showing the name of the person.
AGin the camera can be closed down by pressing 'q' key on the keyboard.
