package raymond;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;

public class Main {

    public static void main(String[] args) throws Exception {
        // Configuration
        final String testImgPath = "test-images/eclipse.jpg";
        final String trainingDirPath = "training-faces";
        final String modelDirPath = "models";

        // File validation phase
        File testImgFile = new File(testImgPath);
        if (!testImgFile.exists()) {
            throw new IllegalArgumentException(
                "Test image path is invalid"
            );
        }
        File trainingDirFile = new File(trainingDirPath);
        if (!trainingDirFile.isDirectory()) {
            throw new IllegalArgumentException(
                "Training directory path is not a directory"
            );
        }
        BufferedImage testBufImg = ImageIO.read(testImgFile);
        if (testBufImg == null) {
            throw new IOException("Test image could not be read");
        }
        String testImgStem = StringUtils.getStem(testImgFile.getAbsolutePath());
        String testImgFormat = StringUtils.getExtension(testImgFile.getName());

        // Training phase
        List<File> imgList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        Map<Integer, String> labelLegend = FaceTrainer.parseTrainingFaces(
            trainingDirFile.getAbsolutePath(),
            imgList,
            labelList
        );
        FaceRecognizer faceRecognizer = FaceTrainer.createTrainModel(
            imgList,
            labelList
        );
        File saveModelFile = new File(
            modelDirPath + "/face-recognizer.xml"
        );
        // Deletes any old model by the same name just in case
        saveModelFile.delete();
        faceRecognizer.write(saveModelFile.getAbsolutePath());

        // Testing phase
        List<FrameData> roiList = FaceDetector.detectFaces(
            testImgPath,
            "models/haarcascade_frontalface_alt.xml"
        );
        List<FrameData> finalList = identifyFaces(
            testBufImg,
            testImgFormat,
            roiList,
            "models/face-recognizer.xml",
            true
        );
        ImageUtils.drawFrames(testBufImg, finalList, labelLegend, true);
        File outImgFile = new File(
            String.format("%s-out.%s", testImgStem, testImgFormat)
        );
        ImageIO.write(testBufImg, testImgFormat, outImgFile);
        System.out.println(
            "Recognition result image written to: "
            + outImgFile.getAbsolutePath()
        );
    }

    /**
     * Uses the given FaceRecognizer model to evaluate each ROI in the given
     * FrameData list, determining which label each ROI resembles the most as
     * well as a confidence score for each. For each unique label, the ROI with
     * the highest confidence score is saved back into a new FrameData list,
     * which is returned after evaluating all ROIs. <br></br>
     * 
     * @param testImg Test image to predict labels within.
     * @param testImgFormat Format of the test image (e.g. PNG, JPEG).
     * @param roiList List of FrameData objects representing ROIs to evaluate.
     * @param faceRecognizerModelPath Path of the FaceRecognizer model to use.
     * @return A list of FrameData objects, each one representing the unique
     *         best match ROI for a label.
     * @throws IOException For general I/O errors.
     */
    private static List<FrameData> identifyFaces(
        BufferedImage testImg,
        String testImgFormat,
        List<FrameData> roiList,
        String faceRecognizerModelPath
    ) throws IOException {
        return identifyFaces(
            testImg,
            testImgFormat,
            roiList,
            faceRecognizerModelPath,
            false
        );
    }

    /**
     * Underlying method with an extra parameter to enable or disable debugging
     * mode. <br></br>
     * 
     * If <code>debug == true</code>, this method will instead return the
     * original list with all ROIs, where each ROI is modified to include the
     * predicted label and confidence score. <br></br>
     * 
     * See {@link #identifyFaces(BufferedImage, String, List, String)}.
     * 
     * @param testImg Test image to predict labels within.
     * @param testImgFormat Format of the test image (e.g. PNG, JPEG).
     * @param roiList List of FrameData objects representing ROIs to evaluate.
     * @param faceRecognizerModelPath Path of the FaceRecognizer model to use.
     * @param debug Enable or disable debug mode.
     * @return A list of FrameData objects, each one representing the unique
     *         best match ROI for a label.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException For general I/O errors.
     * @throws IllegalArgumentException If the FaceRecognizer model path is
     *                                  invalid.
     */
    private static List<FrameData> identifyFaces(
        BufferedImage testImg,
        String testImgFormat,
        List<FrameData> roiList,
        String faceRecognizerModelPath,
        boolean debug
    ) throws IOException {
        if (testImg == null) {
            throw new NullPointerException("Test image is null");
        }
        if (testImgFormat == null) {
            throw new NullPointerException("Test image format is null");
        }
        if (roiList == null) {
            throw new NullPointerException("ROI list is null");
        }
        if (roiList.isEmpty()) {
            throw new IllegalArgumentException("ROI list is empty");
        }
        if (faceRecognizerModelPath == null) {
            throw new NullPointerException("Face recognizer model path is null");
        }
        Map<Integer, FrameData> labelFrameData = new HashMap<>();
        Map<Integer, Double> labelConfidence = new HashMap<>();
        FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
        try {
            faceRecognizer.read(faceRecognizerModelPath);
        }
        catch (Exception e) {
            throw new IllegalArgumentException(
                "Cannot read FaceRecognizer model: " + faceRecognizerModelPath
            );
        }
        for (FrameData frameData: roiList) {
            BufferedImage roiImg = testImg.getSubimage(
                frameData.xCoord,
                frameData.yCoord,
                frameData.width,
                frameData.height
            );
            File tempRoiFile = File.createTempFile("temp-roi-img", null);
            ImageIO.write(roiImg, testImgFormat, tempRoiFile);
            Mat imgMat = imread(
                tempRoiFile.getAbsolutePath(),
                IMREAD_GRAYSCALE
            );
            Mat resizedImgMat = ImageUtils.squareMat(imgMat, 256);
            imgMat.deallocate();
            tempRoiFile.delete();
            IntPointer labelPtr = new IntPointer(1);
            DoublePointer confidencePtr = new DoublePointer(1);
            faceRecognizer.predict(resizedImgMat, labelPtr, confidencePtr);
            int label = labelPtr.get();
            double confidence = confidencePtr.get();
            resizedImgMat.deallocate();
            labelPtr.deallocate();
            confidencePtr.deallocate();
            if (label < 0) {
                continue;
            }
            frameData.label = label;
            frameData.confidence = confidence;
            // Lower confidence score is better
            if (!labelConfidence.containsKey(label) ||
                    confidence < labelConfidence.get(label)
            ) {
                labelConfidence.put(label, confidence);
                labelFrameData.put(label, frameData);
            }
        }
        if (debug) {
            return roiList;
        }
        List<FrameData> identifiedFaces = new ArrayList<>();
        for (FrameData frameData: labelFrameData.values()) {
            identifiedFaces.add(frameData);
        }
        return identifiedFaces;
    }
}
