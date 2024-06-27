# The OpenCV Facial Recognition App Written in Java
### What is this?
A fun program that can pinpoint and frame known faces on a variety of test images. It uses pre-trained [YuNet face detection](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and [SFace face recognition](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) deep neural network models to perform its job.

It's a work-in-progress of becoming a fully-fledged web application, with [Next.js](https://nextjs.org/) as the frontend and [Spring Boot 2.x](https://spring.io/projects/spring-boot) as the backend.

## Dependencies
- [Maven 3.x](https://maven.apache.org/download.cgi)
- [Node.js 18.17+](https://nodejs.org/en)
- [Java Development Kit 8.x](https://www.oracle.com/java/technologies/downloads/#java8)

## Why Java?
That's a really good question. I started this project because my internship manager asked me to do this, both as a learning experience and as a test of my skills. Before I started looking at OpenCV, I asked him what programming language I should use and the answer was ***"preferably Java"***.

This made sense in the context of the workplace because most of our projects were written in Java. However, this did not make sense in the context of OpenCV as I discovered quickly that OpenCV is native to C/C++. While there is an official OpenCV Java interface, I chose to use a [third-party interface](https://github.com/bytedeco/javacv) because setting up OpenCV libraries was too much of a hassle for me.

The reason I say using Java makes no sense is because there was zero technical reason for me to do this in Java. I'm familiar enough with C++ and Python to have not done this in Java, and it wasn't required by my manager. But from the moment I started writing the code, I gaslighted myself into thinking that I was in too deep to switch gears.

As a result, I dealt with generally poor documentation (having to cross-reference C++ docs), unhelpful error messages (if C++ errors weren't cryptic enough, they look worse coming from the JVM), and a little bit of C pointer memory management. I've also had to ChatGPT almost every OpenCV-related error I ran into because there apparently isn't very strong community support behind OpenCV's face module.

But hey, in the end I have a program that doesn't crash AND can recognize faces with high accuracy.
