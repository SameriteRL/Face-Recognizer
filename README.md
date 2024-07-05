# The OpenCV Face Recognition App Written in Java
### What is this?
A fun web application that can read one or more faces and pinpoint them with high accuracy on any number of subsequent test images. Uses a [Next.js](https://nextjs.org/) frontend and a [Spring Boot](https://spring.io/projects/spring-boot) backend.

Currently, the web interface only allows one target face and one test image to be uploaded but I'm working on that. I'm hoping that in the near future, I'll be able to deploy this app on an actual website.

### How do I use it?
After following setup instructions in the [Dependencies & Setup](#dependencies--setup) section, then with the Next.js and Spring Boot servers running:
1. Visit localhost:3000 where you'll be greeted with a simple, very carefully thought out, well-made interface (thank you Lawrence).
2. Upload an image of your face or some target face that you want the program to pinpoint in the following test image.
3. Upload a test image, which the app will attempt to identify said face within.

The app will spit out a result image, which is the test image with a red bounding box drawn around the target's face. If the target face can't be found in the test image, the test image is just rendered with no modifications.

## Dependencies & Setup
- [Node.js 18.17+](https://nodejs.org/en)
- [Java Development Kit 8.x](https://www.oracle.com/java/technologies/downloads/#java8)
- [Maven 3.x](https://maven.apache.org/download.cgi) (optional)

It's worth mentioning that I use `pnpm` to manage frontend packages and to run the frontend, although I don't know whether it's actually required to run this project.

Install the necessary `node_modules` by running `npm install` inside the frontend Next.js project, or whatever the `pnpm` equivalent is. I run `pnpm dev` to run the frontend server (again, I don't know what I'm doing with frontend development).

For the backend Spring Boot project, running `mvn package` (or `./mvnw package` if you don't have Maven) should be all that's required. After that, `Application.java` can be executed to start up the backend server.

## Credits
This app utilizes pre-trained [YuNet face detection](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace face recognition](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) deep neural network models, which are present in this repository.

Thank you **Professor Shiqi Yu** and **Yuantao Feng** for training and providing the YuNet model!

Thank you **Professor Weihong Deng**, **PhD Candidate Zhong Yaoyao**, and **Master Candidate Chengrui Wang** for training and providing the SFace model!

## Why Java?
I started this project because my internship manager asked me to do this, both as a learning experience and as a test of my skills. Before I started looking at OpenCV, I asked him what programming language I should use and the answer was ***"preferably Java"***.

This made sense in the context of the workplace because most of our projects were written in Java. However, this did not make sense in the context of OpenCV as I discovered quickly that OpenCV is native to C++. While there is an official OpenCV Java API, I chose to use a [third-party API](https://github.com/bytedeco/javacv) mostly because the official one isn't available as a Maven dependency.

The reason I say using Java makes no sense is because there was zero technical reason for me to do so. I'm familiar enough with C++ and Python to have not done this in Java, and it wasn't required by my manager. But ever since I started writing the code, I've been gaslighting myself into thinking that I was in too deep already.

As a result, I dealt with scarce documentation (having to cross-reference C++ docs), unhelpful error messages (if C++ errors weren't cryptic enough, they look worse coming from the JVM), and a LOT of manual memory management (why do I have to free C pointers in Java?).

But hey, the program works so the end justifies the means.
