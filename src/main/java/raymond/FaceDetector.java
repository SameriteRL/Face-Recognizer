package raymond;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;
import java.io.IOException;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;

/**
 * Utilizes the YuNet deep neural network face detection model. Thank you
 * Professor Shiqi Yu and Yuantao Feng! <br></br>
 * 
 * https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
 * <br></br>
 * 
 * Note that not all image formats are supported for facial recognition due to
 * limitations of the <code>cv::imread()</code> function. You can find a list
 * of supported formats here: <br></br>
 * 
 * https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
 */
public class FaceDetector {

    /**
     * Performs facial detection on an image and returns a Mat representing
     * detected faces and their corresponding coordinates/dimensions. Returns
     * an empty Mat (a Mat with 0 rows) if no faces are detected. <br></br>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgPath Path of the image to detect faces from.
     * @param detectorModelPath Path of the YuNet face detector model.
     * @return A Mat representing detected faces in the image.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector path is invalid.
     * @throws IOException If the image is empty or invalid, or for general I/O
     *                     errors.
     */
    public static Mat detectFaces(
        String imgPath,
        String detectorModelPath
    ) throws IOException {
        if (imgPath == null) {
            throw new NullPointerException("Image path");
        }
        Mat imgMat = imread(imgPath);
        try {
            return detectFaces(imgMat, detectorModelPath);
        }
        // Remember kids, always free memory before crashing and burning
        catch (Exception e) {
            imgMat.deallocate();
            throw e;
        }
    }

    /**
     * Performs facial detection on an image and returns a Mat representing
     * detected faces and their corresponding coordinates/dimensions. Returns
     * an empty Mat (a Mat with 0 rows) if no faces are detected. <br></br>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgMat Image Mat to detect faces from.
     * @param detectorModelPath Path of the YuNet face detector model.
     * @return A Mat representing detected faces in the image.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector path is invalid.
     * @throws IOException If the image is empty or invalid, or for general I/O
     *                     errors.
     */
    public static Mat detectFaces(
        Mat imgMat,
        String detectorModelPath
    ) throws IOException {
        if (imgMat == null) {
            throw new NullPointerException("Image Mat");
        }
        if (detectorModelPath == null) {
            throw new NullPointerException("Detector model path");
        }
        if (imgMat.data() == null || imgMat.rows() <= 0 || imgMat.cols() <= 0) {
            throw new IOException("Invalid image Mat");
        }
        File detectorModelFile = new File(detectorModelPath);
        if (!detectorModelFile.exists() || detectorModelFile.isDirectory()) {
            throw new IllegalArgumentException("Invalid face detector model path");
        }
        Mat facesMat = new Mat();
        Size detectorSize = new Size(imgMat.cols(), imgMat.rows());
        try (FaceDetectorYN detector =
                FaceDetectorYN.create(detectorModelPath, "", detectorSize)
        ) {
            detector.detect(imgMat, facesMat);
        }
        catch (Exception e) {
            facesMat.deallocate();
            detectorSize.deallocate();
            throw e;
        }
        detectorSize.deallocate();
        return facesMat;
    }
}
