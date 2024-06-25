# The OpenCV Facial Recognition Program Written in Java
### What is this?
A fun program that can be trained with images of two or more persons, then pinpoint and frame those faces on subsequent test images.

## Project Breakdown And Usage (NEEDS UPDATING)
### Dependencies
This project uses [Maven 3.x](https://maven.apache.org/download.cgi) for dependency management, so make sure that's installed before doing anything else. All dependencies are listed in the repository's `pom.xml`, and imports should be automatically resolved as long as Maven is working.

### Description of source code files

`Main.java` - Main top-level class that utilizes all other classes. This is the entry point of the program. \
`FaceTrainer.java` - Creates and trains the facial recognition model using user-provided faces, and returns it as a Java object. \
`FaceDetector.java` - Performs facial detection and returns the coordinates of all regions of interest (ROIs) to be processed by the face recognizer. \
`FrameData.java` - Simple class that works as a collection of data. \
`ImageUtils.java` - Utility class containing image manipulation methods that aren't closely related ot the main facial detection/recognition logic (e.g. drawing frames onto the image). \
`StringUtils.java` - Utility class containing miscellaneous string observer methods, particularly for paths.

### General program flow

1. Create and train a facial recognition model using user-provided samples from a user-provided directory. Each subdirectory (subject) is assigned a unique label.
2. Write the model to a file inside a user-provided directory.
3. Use a pre-trained Haar cascade classifier model to detect all ROIs in a user-provided test image.
4. Use the model trained in step 1 to predict the label and confidence score associated with each ROI. This project uses the [Fisherfaces algorithm](https://docs.opencv.org/4.x/da/d60/tutorial_face_main.html#tutorial_face_fisherfaces).
5. Draw frames onto each ROI in the test image and output it as a new image inside a user-provided directory.

At the top of the main method in `Main.java`, you can configure input/output paths as you like.

### Training configuration (required)
The directory for training samples must be structured as follows. Note that there **must be two or more subjects present**, as required by OpenCV's Fisherfaces and Eigenfaces models. Training samples are automatically converted to grayscale and made square before training, for ease of use. Any empty directories or non-supported images will be ignored. See the docs on [`cv::imread()`](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8) for details on supported images.
```
training-faces
|-- person1
|   |-- photo1.png
|   `-- photo2.jpg
`-- person2
|   |--photo1.pgm
|   `--photo2.ppm
...
```

Make sure the `trainingDirPath` variable in `Main.java` is correctly set to your local training directory. Other than that, make sure `testImgPath` is valid, otherwise the program will throw an error. `modelDirPath` and `outputDirPath` will be created automatically if they don't already exist.

### Face detection configuration (optional)
For facial detection, the Alt Haar Cascade Frontal Face model is used by default (`haarcascade_frontalface_alt.xml`) but you can use a different one if you want. There are a few others in the repository's `models` directory and sample images processed by each model can be found in `models/haar-cascade-demos`. To switch models, modify the call to `FaceDetector.detectFaces()` in the main method.

Once you've configured everything as described above, you should be able to run `Main.java` and produce results. If not, please read the code and Javadocs (or womp womp if you can't read the code).

## Why Java?
That's a really good question. I started this project because my internship manager asked me to do this, both as a learning experience and as a test of my skills. Before I started looking at OpenCV, I asked him what programming language I should use and the answer was ***"preferably Java"***.

This made sense in the context of the workplace because most of our projects were written in Java. However, this did not make sense in the context of OpenCV as I discovered quickly that OpenCV is native to C/C++. While there is an official OpenCV Java interface, I chose to use a [third-party interface](https://github.com/bytedeco/javacv) because downloading OpenCV libraries and setting them up manually was too much of a hassle for me.

The reason I say using Java makes no sense is because technically speaking there was zero reason for me to do this in Java. I'm familiar enough with C++ and Python to have not done this in Java, and it wasn't required by my manager. But from the moment I started writing the code, I gaslighted myself into thinking that I was in too deep to switch gears.

As a result, I dealt with generally poor documentation (having to cross-reference C++ docs), unhelpful error messages (if C++ errors weren't cryptic enough, they look worse coming from the JVM), and a little bit of C pointer memory management. I've also had to ChatGPT almost every OpenCV-related error I ran into because there apparently isn't very strong community support behind OpenCV's face module.

But hey, in the end I have a program that doesn't crash AND can recognize faces to an arbitrary degree of accuracy.
