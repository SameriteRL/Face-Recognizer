package raymond;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import raymond.classes.ROIData;
import raymond.service.FacePredictorService;
import raymond.service.FileService;
import raymond.service.ImageService;
import raymond.utils.StringUtils;

@SpringBootTest
class FaceRecognizerAppTest {

    @Autowired
    public FacePredictorService facePredictor;

    @Autowired
    public ImageService imageService;

    @Autowired
    public FileService fileService;

    @Test
    public void test() throws Exception {
        final String testImgPath = "test_images/skating.jpg";
        final String knownFacesDirPath = "known_faces";
        final String detectorModelPath =
            fileService.getResourceFile("models/yunet_detection_2023mar.onnx")
                       .getAbsolutePath();
        final String recognizerModelPath = 
            fileService.getResourceFile("models/sface_recognition_2021dec.onnx")
                       .getAbsolutePath();
        final String testImgFormat = StringUtils.getExtension(testImgPath);
        Map<String, List<Mat>> knownFaces = facePredictor.parseKnownFaces(
            knownFacesDirPath,
            detectorModelPath,
            recognizerModelPath
        );
        List<ROIData> roiList = null;
        try (FaceRecognizerSF faceRecognizer = 
                FaceRecognizerSF.create(recognizerModelPath, "")
        ) {
            roiList = facePredictor.predictFaces(
                testImgPath,
                knownFaces,
                detectorModelPath,
                faceRecognizer
            );
        }
        BufferedImage bufImg = imageService.createBufferedImage(testImgPath);
        imageService.drawFrames(bufImg, roiList, true);
        File outImgFile = new File("out." + testImgFormat);
        ImageIO.write(bufImg, testImgFormat, outImgFile);
    }
}
