package raymond.service;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_core.vconcat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Objects;

import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

/**
 * Utilizes the YuNet deep neural network face detection model. Thank you
 * Professor Shiqi Yu and Yuantao Feng! <p>
 * 
 * https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
 * <p>
 * 
 * Note that not all image formats are supported for facial recognition due to
 * limitations of the {@code cv::imread()} function. You can find a list
 * of supported formats here: <p>
 * 
 * https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
 */
@Service
public class FaceDetectorService {

    @Autowired
    private MatService matService;

    /**
     * Allocates and returns a Mat representing face box data after performing
     * face detection on an image. <p>
     * 
     * It is the caller's responsibility to properly deallocate the
     * returned Mat. <p>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgBytes Byte array of the image to detect faces from.
     * @param fd YuNet face detection model.
     * @return See {@link #detectFaces(Mat, String)}
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector path is invalid or
     *                                  the image is empty or invalid.
     * @throws IOException For general I/O errors.
     */
    public Mat detectFaces(
        byte[] imgBytes,
        FaceDetectorYN fd
    ) throws IOException {
        Objects.requireNonNull(imgBytes, "Image byte array");
        Objects.requireNonNull(fd, "Face detector model");
        File tempInputFile = File.createTempFile("inputImg", null);
        try (OutputStream fileOutStream = new FileOutputStream(tempInputFile)) {
            fileOutStream.write(imgBytes);
            fileOutStream.flush();
            return detectFaces(tempInputFile.getAbsolutePath(), fd);
        }
        finally {
            if (tempInputFile != null) {
                tempInputFile.delete();
            }
        }
    }

    /**
     * Allocates and returns a Mat representing face box data after performing
     * face detection on an image. <p>
     * 
     * It is the caller's responsibility to properly deallocate the
     * returned Mat. <p>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgPath Path of the image to detect faces from.
     * @param fd YuNet face detection model.
     * @return See {@link #detectFaces(Mat, String)}
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector path is invalid or
     *                                  the image is empty or invalid.
     */
    public Mat detectFaces(String imgPath, FaceDetectorYN fd) {
        Objects.requireNonNull(imgPath, "Image path");
        Objects.requireNonNull(fd, "Face detector model");
        Mat imgMat = null;
        try {
            imgMat = imread(imgPath);
            return detectFaces(imgMat, fd);
        }
        finally {
            if (imgMat != null) {
                imgMat.close();
            }
        }
    }

    /**
     * Allocates and returns a Mat representing face box data after performing
     * face detection on an image. <p>
     * 
     * The original Mat is not modified or deallocated as a result of the
     * operation; it is the responsibility of the caller to later deallocate it
     * as well as the returned Mat. <p>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgMat Image Mat to detect faces from.
     * @param fd YuNet face detection model.
     * @return A 2D Mat of shape [num_faces, 15]
     *         <ul>
     *           <li> 0-1:   x, y of bounding box top left corner
     *           <li> 2-3:   width, height of bbox
     *           <li> 4-5:   x, y of right eye
     *           <li> 6-7:   x, y of left eye
     *           <li> 8-9:   x, y of nose tip
     *           <li> 10-11: x, y of right corner of mouth
     *           <li> 12-13: x, y of left corner of mouth
     *           <li> 14:    face score
     *         </ul>
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector path is invalid or
     *                                  the image is empty or invalid.
     */
    public Mat detectFaces(Mat imgMat, FaceDetectorYN fd) {
        Objects.requireNonNull(imgMat, "Image Mat");
        Objects.requireNonNull(fd, "Face detector model");
        if (imgMat.empty()) {
            throw new IllegalArgumentException("Invalid image Mat");
        }
        Mat aggResult = new Mat(0, 15, 5),
            tempImgMat = imgMat.clone(),
            tempDetectResult = null,
            tempConcatResult = null,
            tempResizedImg = null;
        double scaleFactor = 1.0;
        do {
            try {
                try (Size inputSize =
                        new Size(tempImgMat.cols(), tempImgMat.rows())
                ) {
                    fd.setInputSize(inputSize);
                }
                tempDetectResult = new Mat();
                fd.detect(tempImgMat, tempDetectResult);
                if (tempDetectResult.rows() > 0) {
                    // Maps face detection results to original image
                    if (scaleFactor != 1.0) {
                        try (FloatRawIndexer indexer =
                                tempDetectResult.createIndexer()
                        ) {
                            for (int i = 0; i < tempDetectResult.rows(); ++i) {
                                for (int j = 0; j < 14; ++j) {
                                    indexer.put(
                                        i,
                                        j,
                                        Math.round(
                                            indexer.get(i, j)
                                            / scaleFactor
                                        )
                                    );
                                }
                            }
                        }
                    }
                    // Aggregates face detection results
                    tempConcatResult = new Mat();
                    vconcat(aggResult, tempDetectResult, tempConcatResult);
                    aggResult.close();
                    aggResult = tempConcatResult;
                }
                scaleFactor -= 0.2;
                tempResizedImg = matService.resizeImg(imgMat, scaleFactor);
            }
            catch (Exception e) {
                if (aggResult != null) {
                    aggResult.close();
                }
                throw e;
            }
            finally {
                if (tempDetectResult != null) {
                    tempDetectResult.close();
                }
                if (tempImgMat != null) {
                    tempImgMat.close();
                }
                tempImgMat = tempResizedImg;
            }
        } while (imgMat.cols() > 300
                    && imgMat.rows() > 300
                    && scaleFactor > 0.3);
        return aggResult;
    }
}
