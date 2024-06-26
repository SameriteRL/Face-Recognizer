package raymond;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

public class Main {

    public static void main(String[] args) throws Exception {
        // Configuration
        final String testImgPath = "./test-faces/cavemen.jpg";
        final String facesDirPath = "./known-faces";
        final String detectorModelPath = "./models/yunet_detection_2023mar.onnx";
        final String recognizerModelPath = "./models/sface_recognition_2021dec.onnx";
        final String outputDirPath = "./output";

        // File validation
        File testImgFile = new File(testImgPath);
        if (!testImgFile.exists()) {
            throw new IllegalArgumentException("Invalid test image path");
        }
        File trainingDirFile = new File(facesDirPath);
        if (!trainingDirFile.isDirectory()) {
            throw new IllegalArgumentException(
                "Faces directory path does not exist or is not a directory"
            );
        }
        // Ensures output directory exists before writing to it
        File outputDirFile = new File(outputDirPath);
        outputDirFile.mkdir();
        BufferedImage testBufImg = ImageIO.read(testImgFile);
        if (testBufImg == null) {
            throw new IOException("Test image could not be read");
        }
        String testImgStem = StringUtils.getStem(testImgFile.getName());
        String testImgFormat = StringUtils.getExtension(testImgFile.getName());

        // Face prediction phase
        List<FrameData> roiList = FacePredictor.predictFaces(
            testImgPath,
            facesDirPath,
            detectorModelPath,
            recognizerModelPath
        );
        ImageUtils.drawFrames(testBufImg, roiList);
        File outImgFile = new File(
            String.format(
                "%s/%s-out.%s",
                outputDirFile.getAbsolutePath(),
                testImgStem,
                testImgFormat
            )
        );
        ImageIO.write(testBufImg, testImgFormat, outImgFile);
        System.out.println(
            "Recognition result image written to: "
            + outImgFile.getAbsolutePath()
        );
    }
}
