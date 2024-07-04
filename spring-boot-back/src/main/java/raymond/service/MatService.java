package raymond.service;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.springframework.stereotype.Service;

@Service
public class MatService {

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
     * @throws IllegalArgumentException If either specified width or height is
     *                                  not a positive value.
     */
    public Mat resizeImg(Mat src, int width, int height) {
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
        catch (Exception e) {
            if (resizedImg != null) {
                resizedImg.deallocate();
            }
            throw e;
        }
        return resizedImg;
    }

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
     * @throws IllegalArgumentException If the specified scale is not a
     *                                  positive value.
     */
    public Mat resizeImg(Mat src, double scale) {
        if (scale <= 0) {
            throw new IllegalArgumentException("Scale must be a positive value");
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
        catch (Exception e) {
            if (resizedImg != null) {
                resizedImg.deallocate();
            }
            throw e;
        }
        return resizedImg;
    }

    /**
     * 
     * @param fd YuNet face detection model.
     * @param fr SFace face recognition model.
     * @param srcMat Source image Mat.
     * @param detectResult A Mat containing face detection results from the
     *                     YuNet model.
     * @return A list of face feature Mats.
     * @throws IOException If the image is empty or invalid, or for general
     *                     I/O errors.
     */
    public List<Mat> getFeatureMats(
        FaceDetectorYN fd,
        FaceRecognizerSF fr,
        Mat srcMat,
        Mat detectResult
    ) throws IOException {
        List<Mat> featureMatList = new ArrayList<>();
        try {
            // Ideally one person per image, but multiple instances
            // of the same person will still work.
            for (int i = 0; i < detectResult.rows(); ++i) {
                Mat featureMat = getFeatureMat(
                    fr,
                    srcMat,
                    detectResult.row(i)
                );
                featureMatList.add(featureMat);
            }
        }
        finally {
            if (detectResult != null) {
                detectResult.deallocate();
                detectResult = null;
            }
        }
        return featureMatList;
    }

    /**
     * Allocates and returns a face feature Mat of an image. Requires a SFace
     * face recognizer as well as a Mat of face box coordinates and dimensions,
     * typically determined using the YuNet face detection model. <p>
     * 
     * The original Mat is not modified or deallocated as a result of the
     * operation; it is the responsibility of the caller to later deallocate it
     * as well as the returned Mat.
     * 
     * @param recognizerModelPath Path of the SFace face recognizer model.
     * @param srcImg Source image, typically the whole original image.
     * @param fBox Face box coordinates and dimensions on the source image,
     *             typically one row of a Mat.
     * @return The feature Mat of the image.
     */
    public Mat getFeatureMat(
        String recognizerModelPath,
        Mat srcImg,
        Mat faceBox
    ) {
        if (recognizerModelPath == null) {
            throw new NullPointerException("Face recognizer model path");
        }
        try (FaceRecognizerSF fr =
                FaceRecognizerSF.create(recognizerModelPath, "")
        ) {
            return getFeatureMat(fr, srcImg, faceBox);
        }
    }

    /**
     * Allocates and returns a face feature Mat of an image. Requires a SFace
     * face recognizer as well as a Mat of face box coordinates and dimensions,
     * typically determined using the YuNet face detection model. <p>
     * 
     * The original Mat is not modified or deallocated as a result of the
     * operation; it is the responsibility of the caller to later deallocate it
     * as well as the returned Mat.
     * 
     * @param fr SFace face recognizer model.
     * @param srcImg Source image, typically the whole original image.
     * @param fBox Face box coordinates and dimensions on the source image,
     *             typically one row of a Mat.
     * @return The feature Mat of the image.
     */
    public Mat getFeatureMat(
        FaceRecognizerSF fr,
        Mat srcImg,
        Mat faceBox
    ) {
        if (fr == null) {
            throw new NullPointerException("Face recognizer model");
        }
        if (srcImg == null) {
            throw new NullPointerException("Source image Mat");
        }
        if (faceBox == null) {
            throw new NullPointerException("Face box Mat");
        }
        Mat alignedMat = null, featureMat = null;
        try {
            alignedMat = new Mat();
            featureMat = new Mat();
            fr.alignCrop(srcImg, faceBox, alignedMat);
            fr.feature(alignedMat, featureMat);
            // Don't know why this has to be cloned, it just does
            Mat featureMatClone = featureMat.clone();
            return featureMatClone;
        }
        finally {
            if (alignedMat != null) {
                alignedMat.deallocate();
            }
            if (featureMat != null) {
                featureMat.deallocate();
            }
        }
    }
}
