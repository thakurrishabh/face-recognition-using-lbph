# face-recognition-using-lbph

This project I've done as part of my own interest.

Technology used: Python, Opencv, LBPH, viola jones face detection classifier

To perform LBPH I've created a module lbph.py which contains the functions required to perform lbph.

Introduction and Working:
Face recognition is performed using local binary pattern histograms(lbph) and face detection has been done using viola jones haar cascade classifier. The user has to first run facerecognition.py to register his face by selecting 1 to train the algorithm followed by entering password and name. The algorithm takes 64 images of the user and stores in a folder called database. The algorithm then runs LBPH on the images and stores the trained classifier in trainedRec.py. The user can then select recognize to test whether he is an authorized person or not. The threshold posed by the condition if (prediction[1])<100 should be adjusted to adjust the sensitivity of face recognition.

Local Binary Pattern Histograms:
Suppose we have a facial image in grayscale.
We can get part of this image as a window of 3x3 pixels.
It can also be represented as a 3x3 matrix containing the intensity of each pixel (0~255).
Then, we need to take the central value of the matrix to be used as the threshold.
This value will be used to define the new values from the 8 neighbors.
For each neighbor of the central value (threshold), we set a new binary value. We set 1 for values equal or higher than the threshold and 0 for values lower than the threshold.
Now, the matrix will contain only binary values (ignoring the central value). We need to concatenate each binary value from each position from the matrix line by line into a new binary value (e.g. 10001101). Note: some people use other approaches to concatenate the binary values (e.g. clockwise direction), but the final result will be the same.
Then, we convert this binary value to a decimal value and set it to the central value of the matrix, which is actually a pixel from the original image.
At the end of this procedure (LBP procedure), we have a new image which represents better the characteristics of the original image.
Now, using the image generated in the last step, we can divide the image into multiple grids

Based on the image above, we can extract the histogram of each region as follows:
As we have an image in grayscale, each histogram (from each grid) will contain only 256 positions (0~255) representing the occurrences of each pixel intensity.
Then, we need to concatenate each histogram to create a new and bigger histogram. Supposing we have 8x8 grids, we will have 8x8x256=16.384 positions in the final histogram. The final histogram represents the characteristics of the image original image.
