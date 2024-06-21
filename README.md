# The OpenCV Facial Recognition Program Written Entirely in Java
### What is this?
A fun program that can be trained with images of one or more person(s) and pinpoint + frame those faces on subsequent test images.

### Why Java?
That's a really good question. I started this project because my internship manager asked me to do this, both as a learning experience and as a test of my skills. Without doing any research or planning beforehand, I asked him what programming language I should use and the answer was ***"preferably Java"***.

This made sense in the context of the workplace because most of our projects were written in Java. This did not make sense, however, in the context of OpenCV as I found out quickly that OpenCV is native to C/C++. While there is an official OpenCV Java interface, I chose to use a third-party interface because setting up OpenCV was too much of a hassle for me.

The reason I say using Java makes no sense is because technically speaking, there was almost zero reason for me to do this in Java. I'm familiar enough with C++ and Python to have not done this in Java, and it wasn't even required by my manager. But from the moment I started writing the code, I gaslighted myself into thinking that I was in too deep to switch gears.

Because of my decision, I've dealt with a major lack of documentation (I had to cross-reference the C++ docs), unhelpful error messages (if C++ errors weren't cryptic enough as is, they look even worse coming out of the JVM), and a little bit of C pointer memory management. I've caused stack overflow errors from both Java and C++, and I've had to ChatGPT almost every OpenCV-related error I ran into because there apparently isn't very strong community support behind OpenCV's face module.

But hey, now I have a program that doesn't crash AND can recognize faces to an arbitrary degree of accuracy.

## Quick Start
This project uses Maven 3.x. That's all for now, I'll probably add more here later.
