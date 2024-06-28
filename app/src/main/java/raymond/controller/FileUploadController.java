package raymond.controller;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import raymond.classes.FaceBox;
import raymond.service.FacePredictorService;
import raymond.service.FileService;
import raymond.service.ImageService;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

@RestController
public class FileUploadController {

    @Autowired
    private FileService fileService;

    @Autowired
    private FacePredictorService facePredictor;

    @Autowired
    private ImageService imageService;

    @PostMapping("/submit")
    public ResponseEntity<byte[]> handleImageSubmit(
        @RequestParam("faceImg") MultipartFile faceImg,
        @RequestParam("testImg") MultipartFile testImg
    ) throws IOException {
        final String detectorModelPath =
            fileService.getResourceFile("models/yunet_detection_2023mar.onnx")
                       .getAbsolutePath();
        final String recognizerModelPath = 
            fileService.getResourceFile("models/sface_recognition_2021dec.onnx")
                       .getAbsolutePath();
        Path tempFacesPath = null;
        File testImgFile = null;
        try {
            tempFacesPath = Files.createTempDirectory("known-faces");
            Path faceDirPath = Files.createDirectories(tempFacesPath.resolve("face1"));
            Path faceImgPath = Files.createTempFile(faceDirPath, "face", ".jpg");
            try (FileOutputStream fostr = new FileOutputStream(faceImgPath.toFile())) {
                fostr.write(faceImg.getBytes());
            }
            testImgFile = File.createTempFile("testImg", ".jpg");
            try (FileOutputStream fostr = new FileOutputStream(testImgFile)) {
                fostr.write(testImg.getBytes());
            }
            Map<String, List<Mat>> knownFaces = facePredictor.parseKnownFaces(
                tempFacesPath.toAbsolutePath().toString(),
                detectorModelPath,
                recognizerModelPath
            );
            List<FaceBox> roiList = null;
            try (FaceRecognizerSF faceRecognizer = 
                    FaceRecognizerSF.create(recognizerModelPath, "")
            ) {
                roiList = facePredictor.predictFaces(
                    testImgFile.getAbsolutePath(),
                    knownFaces,
                    detectorModelPath,
                    faceRecognizer
                );
            }
            return ResponseEntity.ok(
                imageService.drawBoxesOnImage(
                    testImgFile.getAbsolutePath(),
                    roiList
                )
            );
        }
        catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(new byte[0]);
        }
        finally {
            if (tempFacesPath != null) {
                fileService.deleteDirRecursive(tempFacesPath);
            }
            if (testImgFile != null) {
                testImgFile.delete();
            }
        }
    }
}
