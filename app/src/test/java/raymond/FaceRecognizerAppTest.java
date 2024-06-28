package raymond;

import java.util.List;
import java.util.Map;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import raymond.classes.FaceBox;
import raymond.service.FacePredictorService;
import raymond.service.FileService;
import raymond.service.ImageService;

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
        // final String testImgPath = "test_images/skating.jpg";
        // final String knownFacesDirPath = "known-faces";
        // final String detectorModelPath =
        //     fileService.getResourceFile("models/yunet_detection_2023mar.onnx")
        //                .getAbsolutePath();
        // final String recognizerModelPath = 
        //     fileService.getResourceFile("models/sface_recognition_2021dec.onnx")
        //                .getAbsolutePath();
        // Map<String, List<Mat>> knownFaces = facePredictor.parseKnownFaces(
        //     knownFacesDirPath,
        //     detectorModelPath,
        //     recognizerModelPath
        // );
        // List<FaceBox> roiList = null;
        // try (FaceRecognizerSF faceRecognizer = 
        //         FaceRecognizerSF.create(recognizerModelPath, "")
        // ) {
        //     roiList = facePredictor.predictFaces(
        //         testImgPath,
        //         knownFaces,
        //         detectorModelPath,
        //         faceRecognizer
        //     );
        // }
        // byte[] imgBytes = imageService.drawBoxesOnImage(testImgPath, roiList);
    }
}
