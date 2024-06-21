package raymond;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

public class FaceDetector {

    /**
     * Performs facial detection on an image and returns a list of FrameData
     * objects, where each one represents a region of interest (ROI). Returns
     * an empty list if no ROIs are detected. <br></br>
     * 
     * Uses the Default Haar Cascade Frontal-Face Model to detect faces.
     * 
     * @param imgPath Path of the image to extract ROIs from.
     * @return A list of FrameData objects, each one representing a ROI.
     * @throws NullPointerException If the image path is null.
     * @throws IOException If the image path is invalid, or for general I/O
     *                     errors.
     */
    public static List<FrameData> detectFaces(String imgPath) throws IOException {
        return detectFaces(
            imgPath,
            "models/haarcascade_frontalface_default.xml"
        );
    }

    /**
     * Underlying method with an extra parameter to specify the cascade
     * classifier model to use for face detection. <br></br>
     * 
     * See {@link #detectFaces(String)}
     * 
     * @param imgPath Path of the image to extract ROIs from.
     * @param cascadeModelPath Path of the cascade model to use.
     * @return A list of FrameData objects, each one representing a ROI.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If the image or model paths are invalid, or for
     *                     general I/O errors.
     */
    public static List<FrameData> detectFaces(
        String imgPath,
        String cascadeModelPath
    ) throws IOException {
        if (imgPath == null) {
            throw new NullPointerException("Image path is null");
        }
        if (cascadeModelPath == null) {
            throw new NullPointerException("Cascade model path is null");
        }
        File cascadeModelFile = new File(cascadeModelPath);
        BufferedImage bufImg = ImageUtils.createBufferedImage(imgPath);
        if (bufImg == null) {
            throw new IOException("Image path is invalid or cannot be read");
        }
        if (!cascadeModelFile.exists() || cascadeModelFile.isDirectory()) {
            throw new IOException("Cascade model path is invalid");
        }
        // Face detection
        Mat imgMat = imread(imgPath);
        RectVector detectedFaces = new RectVector();
        try (CascadeClassifier detector =
                new CascadeClassifier(cascadeModelPath)
        ) {
            detector.detectMultiScale(imgMat, detectedFaces);
        }
        catch (Exception e) {
            throw new RuntimeException("Cascade model is invalid");
        }
        List<FrameData> roiData = new ArrayList<>();
        for (int i = 0; i < detectedFaces.size(); ++i) {
            Rect face = detectedFaces.get(i);
            roiData.add(
                new FrameData(
                    face.x(),
                    face.y(),
                    face.width(),
                    face.height()
                )
            );
        }
        return roiData;
    }
}
