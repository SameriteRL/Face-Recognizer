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
     * face detection on an image. Can only detect faces between 10x10px and
     * 300x300px in size, see {@link #detectFaces(Mat, FaceDetectorYN)} for a
     * more flexible solution. <p>
     * 
     * The original Mat is not modified as a result of the operation; it is the
     * responsibility of the caller to later deallocate it as well as the
     * returned Mat. <p>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgMat Image Mat to detect faces in.
     * @param fd YuNet face detection model.
     * @return See {@link #detectFaces(Mat, FaceDetectorYN)}
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector or image is
     *                                  invalid.
     */
    public Mat detectFacesRaw(Mat imgMat, FaceDetectorYN fd) {
        Objects.requireNonNull(imgMat, "Image Mat");
        Objects.requireNonNull(fd, "Face detector model");
        if (imgMat.empty()) {
            throw new IllegalArgumentException("Invalid image Mat");
        }
        if (fd.isNull()) {
            throw new IllegalArgumentException("Face detector is null");
        }
        try (Size inputSize = new Size(imgMat.cols(), imgMat.rows())) {
            fd.setInputSize(inputSize);
        }
        Mat detectResult = new Mat();
        fd.detect(imgMat, detectResult);
        return detectResult;
    }

    /**
     * Allocates and returns a Mat representing face box data after performing
     * face detection on an image. Performs multiple scans at different scales
     * for maximum accuracy, which may result in overlapping bounding boxes.
     * See {@link #detectFacesRaw(Mat, FaceDetectorYN)} for faster results at
     * the cost of constrained input. <p>
     * 
     * It is the caller's responsibility to properly deallocate the
     * returned Mat. <p>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgBytes Byte array of the image to detect faces in.
     * @param fd YuNet face detection model.
     * @return See {@link #detectFaces(Mat, FaceDetectorYN)}
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector or image is
     *                                  invalid.
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
            tempInputFile.delete();
        }
    }

    /**
     * Allocates and returns a Mat representing face box data after performing
     * face detection on an image. Performs multiple scans at different scales
     * for maximum accuracy, which may result in overlapping bounding boxes.
     * See {@link #detectFacesRaw(Mat, FaceDetectorYN)} for faster results at
     * the cost of constrained input. <p>
     * 
     * It is the caller's responsibility to properly deallocate the returned
     * Mat. <p>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgPath Path of the image to detect faces in.
     * @param fd YuNet face detection model.
     * @return See {@link #detectFaces(Mat, FaceDetectorYN)}
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the face detector or image is
     *                                  invalid.
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
     * face detection on an image. Performs multiple scans at different scales
     * for maximum accuracy, which may result in overlapping bounding boxes.
     * See {@link #detectFacesRaw(Mat, FaceDetectorYN)} for faster results at
     * the cost of constrained input. <p>
     * 
     * The original Mat is not modified as a result of the operation; it is the
     * responsibility of the caller to later deallocate it as well as the
     * returned Mat. <p>
     * 
     * Note that not all image formats are supported; see {@link #FaceDetector}
     * for details.
     * 
     * @param imgMat Image Mat to detect faces in.
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
     * @throws IllegalArgumentException If the face detector or image is
     *                                  invalid.
     */
    public Mat detectFaces(Mat imgMat, FaceDetectorYN fd) {
        Objects.requireNonNull(imgMat, "Image Mat");
        Objects.requireNonNull(fd, "Face detector model");
        if (imgMat.empty()) {
            throw new IllegalArgumentException("Invalid image Mat");
        }
        if (fd.isNull()) {
            throw new IllegalArgumentException("Face detector is null");
        }
        Mat aggResult = new Mat(0, 15, 5),
            tmpImg = null,
            tmpConcatResult = null;
        // I hate floating point errors
        int scaleFactor = 10;
        do {
            try (Mat tmpDetectRes = new Mat()) {
                if (tmpImg != null) {
                    tmpImg.close();
                }
                tmpImg = (scaleFactor != 10) ? 
                            matService.resizeImg(imgMat, scaleFactor / 10.0)
                            : imgMat.clone();
                try (Size inputSize = new Size(tmpImg.cols(), tmpImg.rows())) {
                    fd.setInputSize(inputSize);
                }
                fd.detect(tmpImg, tmpDetectRes);
                if (tmpDetectRes.rows() > 0) {
                    if (scaleFactor == 10) {
                        continue;
                    }
                    // Maps face detection results to original image
                    try (FloatRawIndexer indexer = tmpDetectRes.createIndexer()) {
                        for (int i = 0; i < tmpDetectRes.rows(); ++i) {
                            for (int j = 0; j < 14; ++j) {
                                indexer.put(
                                    i,
                                    j,
                                    Math.round(
                                        indexer.get(i, j)
                                        / (scaleFactor / 10.0)
                                    )
                                );
                            }
                        }
                    }
                    // Aggregates face detection results
                    tmpConcatResult = new Mat();
                    vconcat(aggResult, tmpDetectRes, tmpConcatResult);
                    aggResult.close();
                    aggResult = tmpConcatResult;
                }
            }
            catch (Exception e) {
                if (tmpImg != null) {
                    tmpImg.close();
                }
                if (aggResult != null) {
                    aggResult.close();
                }
                throw e;
            }
            finally {
                scaleFactor -= 2;
            }
        } while ((tmpImg.cols() > 300 || tmpImg.rows() > 300)
                    && scaleFactor >= 2);
        return aggResult;
    }
}
