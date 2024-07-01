package raymond.service;

import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import raymond.classes.FaceBox;
import raymond.utils.StringUtils;

@Service
public class FaceRecognizerFacade {

    @Autowired
    private FaceDetectorService faceDetectorService;

    @Autowired
    private FacePredictorService facePredictorService;

    @Autowired
    private ImageService imageService;

    @Autowired
    private FileService fileService;
    
    /**
     * Orchestrates the facial recognition from start to finish, returning the
     * visualized results as a byte array.
     * 
     * @param faceImg Face image to pinpoint in the test image.
     * @param testImg Test image to pinpoint the desired face on.
     * @return A byte array representation of the visualized result.
     * @throws IOException For general I/O errors.
     */
    public byte[] recognizeFaces(
        MultipartFile faceImg,
        MultipartFile testImg
    ) throws IOException {
        String testImgFormat =
            StringUtils.getExtensionWithDot(testImg.getOriginalFilename());
        FaceDetectorYN faceDetector = null;
        FaceRecognizerSF faceRecognizer = null;
        Path tempFacesDirPath = null, testImgPath = null;
        try {
            faceDetector = faceDetectorService.createFaceDetector();
            faceRecognizer = facePredictorService.createFaceRecognizer();
            tempFacesDirPath = Files.createTempDirectory("known-faces");
            Path faceDirPath =
                Files.createDirectories(tempFacesDirPath.resolve("you"));
            Path faceImgPath =
                Files.createTempFile(faceDirPath, null, testImgFormat);
            try (FileOutputStream fostr =
                    new FileOutputStream(faceImgPath.toFile())
            ) {
                fostr.write(faceImg.getBytes());
            }
            testImgPath =
                Files.createTempFile(tempFacesDirPath, null, testImgFormat);
            try (FileOutputStream fostr =
                    new FileOutputStream(testImgPath.toFile())
            ) {
                fostr.write(testImg.getBytes());
            }
            Map<String, List<Mat>> knownFaces =
                facePredictorService.parseKnownFaces(
                    tempFacesDirPath.toString(),
                    faceDetector,
                    faceRecognizer
                );
            List<FaceBox> roiList = null;
            roiList = facePredictorService.predictFaces(
                testImgPath.toString(),
                knownFaces,
                faceDetector,
                faceRecognizer
            );
            return imageService.visualizeBoxes(
                testImgPath.toString(),
                roiList
            );
        }
        finally {
            if (faceDetector != null) {
                faceDetector.deallocate();
            }
            if (faceRecognizer != null) {
                faceRecognizer.deallocate();
            }
            if (tempFacesDirPath != null) {
                fileService.deleteDirRecursive(tempFacesDirPath);
            }
        }
    }
}
