# The OpenCV Facial Recognition App Written in Java
### What is this?
A fun web application that can read one or more faces and pinpoint them with high accuracy on any number of subsequent test images. Uses a [Next.js](https://nextjs.org/) frontend and a [Spring Boot](https://spring.io/projects/spring-boot) backend.

Currently, the web interface only allows one target face and one test image to be uploaded, but I'm working on that. I'm hoping that one day in the near future, I'll be able to deploy this app on an actual website.

### How do I use it?
After following setup instructions in the [Dependencies & Setup](#dependencies--setup) section, then with the Next.js and Spring Boot servers running:
1. Visit [localhost:3000](localhost:3000) where you'll be greeted with a simple, very carefully thought out, well-made interface (thank you Lawrence).
2. Upload an image of your face or some target face that you want the program to pinpoint in the following test image.
3. Upload a test image, which the app will attempt to identify the given face within.

The app will spit out a result image, which is the test image with a red bounding box drawn around the target's face. If the target face can't be found in the test image, the test image is just rendered with no modifications.

## Dependencies & Setup
- [Node.js 18.17+](https://nodejs.org/en)
- [Java Development Kit 8.x](https://www.oracle.com/java/technologies/downloads/#java8)
- [Maven 3.x](https://maven.apache.org/download.cgi)

Also worth mentioning that I use `pnpm` to manage frontend packages and to run the frontend, although I don't know enough about frontend development to know whether it's actually required to run this project.

Install the necessary `node_modules` by running `npm install` inside the frontend Next.js project, or whatever the `pnpm` equivalent is. I run `pnpm dev` to run the frontend server (again, I don't really know what I'm doing).

For the backend Spring Boot project, running `mvn package` should be all that's required. After that, `Application.java` can be executed to start up the backend server.

## Credits
This app utilizes pre-trained [YuNet face detection](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace face recognition](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) deep neural network models, which are already present in this repository.

Thank you **Professor Shiqi Yu** and **Yuantao Feng** for training and providing the YuNet model!

Thank you **Professor Weihong Deng**, **PhD Candidate Zhong Yaoyao**, and **Master Candidate Chengrui Wang** for training and providing the SFace model!

## Why Java?
That's a really good question. I started this project because my internship manager asked me to do this, both as a learning experience and as a test of my skills. Before I started looking at OpenCV, I asked him what programming language I should use and the answer was ***"preferably Java"***.

This made sense in the context of the workplace because most of our projects were written in Java. However, this did not make sense in the context of OpenCV as I discovered quickly that OpenCV is native to C/C++. While there is an official OpenCV Java interface, I chose to use a [third-party interface](https://github.com/bytedeco/javacv) because manually setting up OpenCV libraries was too much of a hassle for me.

The reason I say using Java makes no sense is because there was zero technical reason for me to do this in Java. I'm familiar enough with C++ and Python to have not done this in Java, and it wasn't required by my manager. But from the moment I started writing the code, I gaslighted myself into thinking that I was in too deep to switch gears.

As a result, I dealt with generally poor documentation (having to cross-reference C++ docs), unhelpful error messages (if C++ errors weren't cryptic enough, they look worse coming from the JVM), and more C pointer memory management than I'd like. I've also had to ChatGPT almost every OpenCV-related error I ran into because there apparently isn't very strong community support behind OpenCV's face module.

But hey, the program works so the end justifies the means.
