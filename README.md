## Summary

This project is my Capstone project for the ML-bootcamp (with Alexey Grigorev).
The goal is to gather some experience in various steps of ML pipeline.

I selected the FER2013 *(Facial Expression Recognition)* dataset in order to try solving an `Image Multiclass Classification` problem. Such a problem is interesting because it tries to classify isolated faces among several expressions (7 on this dataset).

Such models can probably be used in:
- the virtual reality or the virtual meetings with avatars (to transcribe the facial expression to the other participants),
- the video games (so that the game can know if it must adjust to the player or not)
- the interactive street advertisements (to evaluate how the public react when exposed to a given product),
- the shows (to evaluate the appreciation of the movie, theater show etc.)
- maybe even to school (so that teachers can know if a kid has a problem, is distracted etc...)
- ...

The full set consists of 35,887 samples, and the data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image.


### Dataset source

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project.
The dataset is too large to be hosted on github, but it can be downloaded from Kaggle:

- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge
- https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz


## Cloud demo

One can try the application here:
https://ml-fer2013.herokuapp.com/input


## Clone repo

```bash
>> git clone https://github.com/Valkea/ML_Facial_Expression_Recognition.git
>> cd ML_Facial_Expression_Recognition
```

## Running jupyter-notebook


```bash
>> jupyter notebook
or
>> jupyter notebook notebook.ipynb
```

*(The notebook menu was created using some HTML, and the GitHub preview doesn't render all of them. But it should work locally)


This project focused on Convolutional Neural Network (CNN) architectures, but several other classifications methods were also tried for comparisons.

The exported model can be tested using:
- the 'Load Models.ipynb' notebook
- the 'test_camera.py' python script (you need a working webcam, but it works with several users at the same time)
- the 'test_image_local.py' python script (just in case you don't have a webcam)
- the 'test_image_remote.py' python script (it sends a picture to the AWS lambda function in order to get a prediction from the remotly hosted model)


## Running locally using python scripts

Install pipenv
```bash
>> pip install pipenv
```

Start the pipenv virtual environment:
```bash
>> pipenv shell
```
*(If it doesnt' work due to the Python 3.8 requirement, you can edit the Pipfile with you own version. I think it should work with any Python3 version has I didn't used any specific function or method.)*

Install the dependencies:
```bash
(venv) >> pipenv install
```

Start Flask development server:
```bash
(venv) >> python fer_server.py
```

Stop with CTRL+C


### Tests
One can check that the server is running by opening the following url:
http://0.0.0.0:5000/input

Then by submitting various predefined pictures, various results should be displayed.

Alternatively a python script can be used to test from 'outside' of the Flask app.
```bash
>> python test_image_local.py
```
This should return an "Happy" label for the given face.

## Docker

### Building a Docker image

```bash
>> docker build -t fer2013-prediction .
```

### Running a local Docker image

```bash
>> docker run -it -p 5000:5000 fer2013-prediction:latest
```

Stop with CTRL+C

Then one can run the same test steps as before... (open input url or run test_image_local.py)

### Pulling a Docker image from Docker-Hub

I pushed a copy of my docker image on the Docker-hub, so one can pull it:

```bash
>> docker pull valkea/fer2013-prediction:latest
```

But this command is optionnal, as running it (see below) will pull it if required.

### Running a Docker image gathered from Docker-Hub

Then the command to start the docker is almost similar to the previous one:

```bash
>> docker run -it -p 5000:5000 valkea/fer2013-prediction:latest
```

Stop with CTRL+C

And once again, one can run the same test steps explained above... (open input url or run test_image_local.py)


## Create a new model file from python script

In order to create a new model .bin file, one can use the following command:

```bash
>> python model_training.py
```
This will use the default input and out names. But this can be changed using the -s (--source) and -d (--destination) parameters.

```bash
>> python model_training.py -s IN_NAME -d OUT_NAME
```

## Cloud deployement

In order to deploy this project, I decided to use Heroku.

So if you don't already have an account, you need to create one and to follow the process explained here: https://devcenter.heroku.com/articles/heroku-cli

Once the Heroku CLI is configured, one can create a project using the following command (or their website):

```bash
>> heroku create ml-fer2013
```

Then, the project can be compiled, published and ran on Heroku, with:

```bash
>> heroku container:push web -a ml-fer2013
>> heroku container:release web -a ml-fer2013
```

Finally, you can open the project url (mine is https://ml-fer2013.herokuapp.com/input), or check the logs using:
```bash
>> heroku logs --tail --app ml-fer2013
```

Finally, here is a great ressource to help deploying projects on Heroku:
https://github.com/nindate/ml-zoomcamp-exercises/blob/main/how-to-use-heroku.md
