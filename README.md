This repository includes a CNN project on image classification.

The model it builds is based on the ResNet architecture with little changes 
on the number of residual blocks and the input shape to meet the image size (28,28,1) of the
**mnist dataset**, thanks to Yan Lecun!

Once the model is trained I test it on the test_set and get around 96% accuracy (open the notebook: Im_Classification.ipynb). 
See you coming, It's not the state of the art, is it? 

I agree with you, but the goal of this project was to build a relatively simple CNN model and then deploy it using **FastAPI** and **Docker**.

That being said, once the model is built I save it using keras and then develop the webserver application using FastAPI as mentioned earlier(see ./app).
This app deploys the model, which means making it available to the end users. Therfore, It provides an interface between the model and the users. 
In our case, it requires users to provide an image of shape of (28, 28, 1), it processes it and rescales it so that its pixels range from
0 to 1 and then gives it to the model as input. The model makes its prediction and returns the **handwriten digit** in the provided image 
as well as the level of confidence. The app takes all of those information and gets them back to the users.

Finally to make sure this works everywhere, I *dockerize* the app, meaning putting the app and all of its dependencies inside a container.
To do that I created a Dockerfile (see ./Dockerfile) to build a docker image which is living at this time in my docker Hub repository.

Enough talking!!! Let's try it out:

To do so, you will first have to open a terminal and follow the steps below. I presume you have Docker already installed! 

- Step 1:

    You will have to clone this repository:

      git clone https://github.com/Segnaly-2021/image_classification_project.git
- Step 2:
  
    The second step is to build the image. One of the simplest ways to do so is to pull the already-existing image living on docker Hub.
    Copy and paste the following line to your terminal:

      docker pull alysegnane/im-classif:modif-ver-resnet
- Step 3:

    Once the image is downloaded, you can run a container out of it. the line below shows how to do that. Copy and paste into your terminal:
  
      docker run --rm -p 8080:8080 alysegnane/im-classif:modif-ver-resnet

- Step 4:

    The server should now be running on your terminal. Leave it there, don't quit the terminal and open your web browser.
    One the most interesting FastAPI features is that it provides a graphical user interface documentation. To visit the one related to our app,
    go visit:
  
      http://localhost:8080/docs

    Once there click on top of the **POST/predict** endpoint on the green bar and more options will become visible.

- Step 5:

    Finally to try  the app out click on **try it out** button on the right corner.
    Submit an image by choosing one from the **images_test** directory. To find it navigate in your local filesystem to find the repository you cloned in **step 1**.
    
    After that, click on execute and let the magic happen!!! 
      
 
