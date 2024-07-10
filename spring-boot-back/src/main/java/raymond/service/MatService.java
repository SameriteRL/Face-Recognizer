package raymond.service;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class MatService {

    @Autowired
    private FaceRecognizerSF fr;

    /**
     * Allocates and returns a new Mat by resizing an image Mat using the
     * specified scale factor. <p>
     * 
     * For example, a scale factor of 0.5 would return an image with half the
     * original width and height, and a factor of 1.5 would return an image
     * with 1.5 times the original width and height. <p>
     * 
     * The original Mat is not modified or deallocated as a result of the
     * operation; it is the responsibility of the caller to later deallocate it
     * as well as the returned Mat.
     * 
     * @param src Source image Mat.
     * @param scale Factor to scale the image by.
     * @return The resized version of the image.
     * @throws NullPointerException If the image Mat is null.
     * @throws IllegalArgumentException If the image Mat is invalid or the
     *                                  scale is a negative value.
     */
    public Mat resizeImg(Mat src, double scale) {
        Objects.requireNonNull(src, "Source image Mat");
        if (src.empty()) {
            throw new IllegalArgumentException("Invalid source image Mat");
        }
        if (scale <= 0) {
            throw new IllegalArgumentException("Scale must be positive");
        }
        if (scale == 1.0) {
            return src.clone();
        }
        Mat resizedImg = null;
        int newWidth = Math.toIntExact(Math.round(src.cols() * scale));
        int newHeight = Math.toIntExact(Math.round(src.rows() * scale));
        try (Size newSize = new Size(newWidth, newHeight)) {
            resizedImg = new Mat(newSize, src.type());
            resize(src, resizedImg, newSize);
        }
        catch (Exception exc) {
            if (resizedImg != null) {
                resizedImg.close();
            }
            throw exc;
        }
        return resizedImg;
    }

    /**
     * Allocates and returns a new Mat by resizing an image Mat to the
     * specified width and height. <p>
     * 
     * The original Mat is not modified or deallocated as a result of the
     * operation; it is the responsibility of the caller to later deallocate it
     * as well as the returned Mat.
     * 
     * @param src Source image Mat.
     * @param width The new desired width.
     * @param height The new desired height.
     * @return The resized version of the image.
     * @throws NullPointerException If the image Mat is null.
     * @throws IllegalArgumentException If the image Mat is invalid or either
     *                                  specified width or height is negative.
     */
    public Mat resizeImg(Mat src, int width, int height) {
        Objects.requireNonNull(src, "Source image Mat");
        if (src.empty()) {
            throw new IllegalArgumentException("Invalid source image Mat");
        }
        if (width <= 0) {
            throw new IllegalArgumentException("New width must be positive");
        }
        if (height <= 0) {
            throw new IllegalArgumentException("New height must be positive");
        }
        if (width == src.cols() && height == src.rows()) {
            return src.clone();
        }
        Mat resizedImg = null;
        try (Size newSize = new Size(width, height)) {
            resizedImg = new Mat(newSize, src.type());
            resize(src, resizedImg, newSize);
        }
        catch (Exception exc) {
            if (resizedImg != null) {
                resizedImg.close();
            }
            throw exc;
        }
        return resizedImg;
    }

    /**
     * Allocates and returns a list of all face feature Mats in an image.
     * Requires a Mat with one or more face boxes determined using the YuNet
     * face detection model. <p>
     * 
     * The given Mats are not modified or deallocated as a result of the
     * operation; it is the responsibility of the caller to later deallocate
     * them as well as the list of returned Mats.
     * 
     * @param srcImg Source image Mat.
     * @param faceBoxes Mat of face boxes determined from the YuNet face
     *                  detection model.
     * @return A list of face feature Mats.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If the image is empty or invalid, or for general
     *                     I/O errors.
     */
    public List<Mat> createFeatureMats(Mat srcImg, Mat faceBoxes) {
        Objects.requireNonNull(srcImg, "Source image Mat");
        Objects.requireNonNull(faceBoxes, "Face boxes Mat");
        List<Mat> featureMatList = new ArrayList<>();
        for (int i = 0; i < faceBoxes.rows(); ++i) {
            featureMatList.add(createFeatureMat(srcImg, faceBoxes.row(i)));
        }
        return featureMatList;
    }

    /**
     * Allocates and returns a face feature Mat of a face in an image. Requires
     * a Mat of the face's bounding box coordinates and dimensions determined
     * using the YuNet face detection model. <p>
     * 
     * The given Mats are not modified or deallocated as a result of the
     * operation; it is the responsibility of the caller to later deallocate
     * them as well as the returned Mat.
     *
     * @param srcImg Source image, typically the whole original image.
     * @param faceBox Mat with one face box determined from the YuNet face
     *                detection model.
     * @return The feature Mat of the image.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the provided face box Mat does not
     *                                  have only one row.
     */
    public Mat createFeatureMat(Mat srcImg, Mat faceBox) {
        Objects.requireNonNull(srcImg, "Source image Mat");
        Objects.requireNonNull(faceBox, "Face box Mat");
        if (srcImg.empty()) {
            throw new IllegalArgumentException("Source image Mat is empty");
        }
        if (faceBox.rows() != 1) {
            throw new IllegalArgumentException("Detect box may only have 1 row");
        }
        try (Mat alignedMat = new Mat(); Mat featureMat = new Mat()) {
            fr.alignCrop(srcImg, faceBox, alignedMat);
            fr.feature(alignedMat, featureMat);
            // Don't know why this has to be cloned, it just does
            Mat featureMatClone = featureMat.clone();
            return featureMatClone;
        }
    }
}
