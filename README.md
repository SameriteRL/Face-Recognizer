# The OpenCV Facial Recognition Program Written in Java
### What is this?
A fun program that can pinpoint and frame known faces on a variety of test images. It uses pre-trained [YuNet face detection](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace face recognition](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) deep neural network models to perform its job.

## Project Breakdown And Usage
### Dependencies
This project uses [Maven 3.x](https://maven.apache.org/download.cgi) for dependency management, so make sure that's installed before doing anything else. All dependencies are listed in the repository's `pom.xml` and imports should be automatically resolved as long as Maven is working.

### Description of source code files

`Main.java` - Top-level class that utilizes all other classes. This is the entry point of the program. \
`FaceDetector.java` - Performs facial detection on images and returns coordinates of regions of interest (ROIs) to be processed by the face recognizer. \
`FacePredictor.java` - Performs facial recognition on an image using ROIs calculated from the face detector. \
`FrameData.java` - Simple class that serves as a collection of data. \
`ImageUtils.java` - Utility class containing image manipulation methods that aren't closely related ot the main facial detection/recognition logic (e.g. drawing frames onto the image). \
`StringUtils.java` - Utility class containing miscellaneous string observer methods, particularly for paths.

### General program flow

1. Use face detection and recognition models to read all user-provided faces from a user-provided faces directory and convert each one into a facial feature matrix. Each image is labeled based on the name of the subdirectory they're located within, and these are the "known" faces that subsequent test faces will be compared against for facial recognition.
2. Repeat step 1 on the test image, parsing all detected faces and converting each into a facial feature matrix.
3. Use the face recognition model to compare all facial feature matrices from the test image against the "known" ones parsed in step 1. A label is assigned to each face based on which known one it resembles the most.
2. Draw frames and labels onto each face in the test image and output it as a new image within a user-provided output directory.

### Face configuration (required)
The required faces directory structure is as follows:
```
known-faces
|-- bobby
|   |-- bobby1.png
|   `-- bobby2.jpg
`-- joey
|   |-- joey1.pgm
|   `-- joey2.ppm
...
```

The requirements for each image are:
- It must contain only the face of the person denoted by the subdirectory name, or multiple instances of that same face.
- The face(s) in the image must be 300x300 pixels or less in size, due to limitations of the YuNet face detector.
- The image must be of a supported format. See the docs on [`cv::imread()`](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gab32ee19e22660912565f8140d0f675a8) for details on supported images.

Any file that's not a subdirectory, is an empty directory, or is an unsupported image will be ignored.

### `Main.java` configuration (required)

Ensure that the `testImgPath` and `facesDirPath` are pointed to the desired test image and known faces directory, respectively. The `detectorModelPath` and `recognizerModelPath` variables should already be set correctly, but if you want to move the models around or rename them, feel free to modify this. Aside from that, `modelDirPath` and `outputDirPath` will be created automatically if they don't already exist.

## Why Java?
That's a really good question. I started this project because my internship manager asked me to do this, both as a learning experience and as a test of my skills. Before I started looking at OpenCV, I asked him what programming language I should use and the answer was ***"preferably Java"***.

This made sense in the context of the workplace because most of our projects were written in Java. However, this did not make sense in the context of OpenCV as I discovered quickly that OpenCV is native to C/C++. While there is an official OpenCV Java interface, I chose to use a [third-party interface](https://github.com/bytedeco/javacv) because setting up OpenCV libraries was too much of a hassle for me.

The reason I say using Java makes no sense is because there was zero technical reason for me to do this in Java. I'm familiar enough with C++ and Python to have not done this in Java, and it wasn't required by my manager. But from the moment I started writing the code, I gaslighted myself into thinking that I was in too deep to switch gears.

As a result, I dealt with generally poor documentation (having to cross-reference C++ docs), unhelpful error messages (if C++ errors weren't cryptic enough, they look worse coming from the JVM), and a little bit of C pointer memory management. I've also had to ChatGPT almost every OpenCV-related error I ran into because there apparently isn't very strong community support behind OpenCV's face module.

But hey, in the end I have a program that doesn't crash AND can recognize faces with high accuracy.
