package raymond;

import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
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
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

public class Main {

    public static void main(String[] args) throws Exception {
        final String testImgPath = "test-faces/eclipse.jpeg";
        File testImgFile = new File(testImgPath);
        if (!testImgFile.exists()) {
            throw new IllegalArgumentException(
                "Test image path is invalid"
            );
        }
        BufferedImage testBufImg = ImageIO.read(testImgFile);
        if (testBufImg == null) {
            throw new IOException("Test image could not be read");
        }
        List<FrameData> roiList =
            FaceDetector.detectFaces(testImgPath, "models/haarcascade_frontalface_alt.xml");
        String testImgName = testImgFile.getCanonicalPath().substring(
            0, testImgFile.getCanonicalPath().lastIndexOf('.')
        );
        String testImgFormat = testImgFile.getName().substring(
            testImgFile.getName().lastIndexOf('.') + 1
        );
        List<FrameData> finalList = identifyFaces(
            testBufImg, testImgFormat, roiList, "models/face-recognizer.xml"
        );
        drawFrames(testBufImg, finalList);
        // drawFrames(testBufImg, roiList); // debug face detection
        File outImgFile = new File(
            String.format("%s-out.%s", testImgName, testImgFormat)
        );
        ImageIO.write(testBufImg, testImgFormat, outImgFile);
    }

    /**
     * Uses the given FaceRecognizer model to evaluate each ROI in the given
     * FrameData list, determining which label each ROI resembles the most as
     * well as a confidence score for each. For each unique label, the ROI with
     * the highest confidence score is saved back into a new FrameData list,
     * which is returned after evaluating all ROIs.
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
        if (roiList == null || roiList.isEmpty()) {
            throw new IllegalArgumentException(
                "ROI list is null or empty"
            );
        }
        Map<Integer, FrameData> labelFrameData = new HashMap<>();
        Map<Integer, Double> labelConfidence = new HashMap<>();
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        faceRecognizer.read(faceRecognizerModelPath);
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
            tempRoiFile.delete();
            IntPointer labelPtr = new IntPointer(1);
            DoublePointer confidencePtr = new DoublePointer(1);
            faceRecognizer.predict(imgMat, labelPtr, confidencePtr);
            int label = labelPtr.get();
            double confidence = confidencePtr.get();
            imgMat.deallocate();
            labelPtr.deallocate();
            confidencePtr.deallocate();
            // Lower confidence score is better
            if (!labelConfidence.containsKey(label) ||
                    confidence < labelConfidence.get(label)
            ) {
                labelConfidence.put(label, confidence);
                labelFrameData.put(label, frameData);
            }
        }
        List<FrameData> identifiedFaceRegions = new ArrayList<>();
        for (FrameData frameData: labelFrameData.values()) {
            identifiedFaceRegions.add(frameData);
        }
        return identifiedFaceRegions;
    }

    /**
     * Draws a red rectangular frame onto the given image for each FrameData in
     * the given list. The image is modified in-place. <br></br>
     * 
     * The X and Y coordinates marks the top left corner of each frame. The
     * width corresponds to the frame's size along the positive X-axis, and the
     * length corresponds to the size along the negative Y-axis.
     * 
     * @param img The image to draw onto.
     * @param frameDataList A list of FrameData objects describing the position
     *                      and dimensions of each frame to be drawn.
     */
    private static BufferedImage drawFrames(
        BufferedImage img,
        List<FrameData> frameDataList
    ) {
        Graphics2D g2d = img.createGraphics();
        g2d.setColor(Color.RED);
        // Stroke and font size proportional to length of img's smallest side
        int length = (img.getWidth() < img.getHeight())
            ? img.getWidth() : img.getHeight();
        int rectStrokeSize = length / 300;
        int fontSize = rectStrokeSize * 15;
        // Dashed line length pattern proportional to stroke size
        // float[] dash = {5, 3, 0.5f, 3};
        // for (int i = 0; i < dash.length; ++i) {
        //     dash[i] = dash[i] * rectStrokeSize;
        // }
        g2d.setStroke(
            new BasicStroke(
                rectStrokeSize,
                BasicStroke.CAP_ROUND,
                BasicStroke.JOIN_ROUND,
                3f,
                null, // dash,
                0f
            )
        );
        g2d.setFont(new Font("Arial", Font.PLAIN, fontSize));
        // int i = 1;
        for (FrameData c: frameDataList) {
            // X and Y coords indicate top left of rectangle
            g2d.drawRect(c.xCoord, c.yCoord, c.width, c.height);
            // Numbers are drawn above the top-middle of rectangle
            // g2d.drawString(
            //     String.valueOf(i),
            //     c.xCoord + c.width / 2 - (fontSize / 3),
            //     c.yCoord - fontSize / 2
            // );
            // ++i;
        }
        g2d.dispose();
        return img;
    }
}
