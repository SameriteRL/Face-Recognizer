package raymond;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;

public class FaceDetector {

    /**
     * Performs facial detection on an image and returns a list of FrameData
     * objects representing regions of interest (ROIs). Returns an empty list
     * if no ROIs are detected. <br></br>
     * 
     * @param imgPath Path of the image to extract ROIs from.
     * @param dnnModelPath Path of the DNN model to use.
     * @return A list of FrameData objects representing ROIs.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If the image or model paths are invalid, or for
     *                     general I/O errors.
     */
    public static List<FrameData> detectFaces(
        String imgPath,
        String dnnModelPath
    ) throws IOException {
        if (imgPath == null) {
            throw new NullPointerException("Image path is null");
        }
        File dnnModelFile = new File(dnnModelPath);
        BufferedImage bufImg = ImageUtils.createBufferedImage(imgPath);
        if (bufImg == null) {
            throw new IOException("Image path is invalid or cannot be read");
        }
        if (!dnnModelFile.exists() || dnnModelFile.isDirectory()) {
            throw new IOException("Face detector model path is invalid");
        }
        // Face detection
        Mat imgMat = imread(imgPath);
        Mat facesMat = new Mat();
        try (FaceDetectorYN detector = FaceDetectorYN.create(
                dnnModelPath, "", new Size(imgMat.cols(), imgMat.rows())
            )
        ) {
            detector.detect(imgMat, facesMat);
        }
        catch (Exception e) {
            e.printStackTrace();
            throw new RuntimeException("Face detector model is invalid");
        }
        List<FrameData> roiData = new ArrayList<>();
        for (int i = 0; i < facesMat.rows(); ++i) {
            roiData.add(new FrameData(i, facesMat));
        }
        return roiData;
    }
}
