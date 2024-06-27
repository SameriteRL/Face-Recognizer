package raymond.service;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;
import org.springframework.stereotype.Service;

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
@Service
public class FaceDetector {

    /**
     * Allocates and returns a Mat containing face detection data after
     * performing face detection on an image. <br></br>
     * 
     * It is the caller's responsibility to properly deallocate the
     * returned Mat. <br></br>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgPath Path of the image to detect faces from.
     * @param detectorModelPath Path of the YuNet face detector model.
     * @return A 2D Mat of shape [num_faces, 15]
     *         <ul>
     *         <li> 0-1:   x, y of bounding box top left corner
     *         <li> 2-3:   width, height of bbox
     *         <li> 4-5:   x, y of right eye
     *         <li> 6-7:   x, y of left eye
     *         <li> 8-9:   x, y of nose tip
     *         <li> 10-11: x, y of right corner of mouth
     *         <li> 12-13: x, y of left corner of mouth
     *         <li> 14:    face score
     *         </ul>
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
        Mat imgMat = null;
        try {
            imgMat = imread(imgPath);
            return detectFaces(imgMat, detectorModelPath);
        }
        finally {
            if (imgMat != null) {
                imgMat.deallocate();
            }
        }
    }

    /**
     * Allocates and returns a Mat containing face detection data after
     * performing face detection on an image. <br></br>
     * 
     * It is the caller's responsibility to properly deallocate the
     * returned Mat. <br></br>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgBytes Byte array of the image to detect faces from.
     * @param detectorModelPath Path of the YuNet face detector model.
     * @return A 2D Mat of shape [num_faces, 15]
     *         <ul>
     *         <li> 0-1:   x, y of bounding box top left corner
     *         <li> 2-3:   width, height of bbox
     *         <li> 4-5:   x, y of right eye
     *         <li> 6-7:   x, y of left eye
     *         <li> 8-9:   x, y of nose tip
     *         <li> 10-11: x, y of right corner of mouth
     *         <li> 12-13: x, y of left corner of mouth
     *         <li> 14:    face score
     *         </ul>
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector path is invalid.
     * @throws IOException If the image is empty or invalid, or for general I/O
     *                     errors.
     */
    public static Mat detectFaces(
        byte[] imgBytes,
        String detectorModelPath
    ) throws IOException {
        if (imgBytes == null) {
            throw new NullPointerException("Image byte array");
        }
        File tempInputFile = null;
        Mat imgMat = null;
        OutputStream fileOutStream = null;
        try {
            tempInputFile = File.createTempFile("tempInputFile", null);
            fileOutStream = new FileOutputStream(tempInputFile);
            fileOutStream.write(imgBytes);
            fileOutStream.flush();
            imgMat = imread(tempInputFile.getAbsolutePath());
            return detectFaces(imgMat, detectorModelPath);
        }
        finally {
            if (tempInputFile != null) {
                tempInputFile.delete();
            }
            if (fileOutStream != null) {
                fileOutStream.close();
            }
            if (imgMat != null) {
                imgMat.deallocate();
            }
        }
    }

    /**
     * Allocates and returns a Mat containing face detection data after
     * performing face detection on an image. <br></br>
     * 
     * It is the caller's responsibility to properly deallocate the
     * returned Mat. <br></br>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgMat Image Mat to detect faces from.
     * @param detectorModelPath Path of the YuNet face detector model.
     * @return A 2D Mat of shape [num_faces, 15]
     *         <ul>
     *         <li> 0-1:   x, y of bounding box top left corner
     *         <li> 2-3:   width, height of bbox
     *         <li> 4-5:   x, y of right eye
     *         <li> 6-7:   x, y of left eye
     *         <li> 8-9:   x, y of nose tip
     *         <li> 10-11: x, y of right corner of mouth
     *         <li> 12-13: x, y of left corner of mouth
     *         <li> 14:    face score
     *         </ul>
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
        if (!detectorModelFile.exists() || !detectorModelFile.isFile()) {
            throw new IllegalArgumentException("Invalid face detector model path");
        }
        Mat facesMat = null;
        Size detectSize = null;
        FaceDetectorYN detector = null;
        try {
            facesMat = new Mat();
            detectSize = new Size(imgMat.cols(), imgMat.rows());
            detector = FaceDetectorYN.create(detectorModelPath, "", detectSize);
            detector.detect(imgMat, facesMat);
        }
        catch (Exception e) {
            if (facesMat != null) {
                facesMat.deallocate();
            }
            throw e;
        }
        finally {
            if (detectSize != null) {
                detectSize.deallocate();
            }
            if (detector != null) {
                detector.deallocate();
            }
        }
        return facesMat;
    }
}
