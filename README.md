# Activity Classification using Motion History Images

## To-do
- Move all options into command line arguments as opposed to hard-coded flags.
- Dynamically fetch the videos used as training data.

## About ##
The following describes a method for recognizing actions in videos. This method relies on computing motion history images (MHIs) from a video, which are representations of motion using the differences between images. We use those representations as input training data to a KNN classifier that can predict the actions (specifically either walking, jogging, running, waving, clapping, or boxing) within similar videos.

## Results ## 
The predictions on the test set had an 88% accuracy. We can notice that the classifier was particularly less effective between similar actions. Waving was most dissimilar to the other actions and was almost completely classified accurately. On the other hand, almost 25% of all instances of jogging were misclassified as either walking or running.

## Instructions
To run:

python3 run.py VIDEO_NAME

It will then output a video file named 'output.mp4' containing the video with the classification on the bottom left corner.

## Recreate results
These will require minor edits in the main function.

### Calculating Hu Moments and Getting Sample Images
The classifier uses pre-calculated Hu moments saved in 'data.npy' and 'labels.npy'.
In order to recreate those files, download each action .zip file from:
https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/

Then save each each collection in a director tree that looks like:
- input
  - boxing
  - handclapping
  - handwaving
  - jogging
  - running
  - walking
  
 Then in the main function, edit the GET_DATA flag to be TRUE and re-run.
 
 ### Finding best K
 The k value for the KNN classifier is hard-coded.
 Edit the TRAIN_MODEL flag to True and re-run to use GridSearchCV to find best k.
 
 ### Generating confusion matrices
 Edit the GENERATE_REPORT flag to be True and re-run.
 
 ### Generating output video from report
 python3 run.py person15_handwaving.avi