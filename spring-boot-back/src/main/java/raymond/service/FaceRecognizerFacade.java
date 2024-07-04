package raymond.service;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

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

    @Autowired
    private MatService matService;

    private static final Set<String> acceptedFormats = new HashSet<String>(
        Arrays.asList(
            "bmp", "dib", "jpeg", "jpg", "jpe", "jp2", "png", "webp",
            "avif", "pbm", "pgm", "ppm", "pxm", "pnm", "pfm", "sr", "ras",
            "tiff", "tif", "exr", "hdr", "pic"
        )
    );

    private static final FilenameFilter imgOnlyFilter = new FilenameFilter() {
        public boolean accept(File file, String name) {
            return acceptedFormats.contains(StringUtils.getExtension(name));
        }
    };
    
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
        Map<Mat, Mat> detectResultFaceFeature = null;
        Map<String, List<Mat>> knownFaces = null;
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
            Mat testImgMat = imread(testImgPath.toString());
            detectResultFaceFeature = new HashMap<>();
            Mat testDetectResult =
                faceDetectorService.detectFaces(testImgMat, faceDetector);
            for (int i = 0; i < testDetectResult.rows(); ++i) {
                Mat testDetectResultRow = testDetectResult.row(i);
                detectResultFaceFeature.put(
                    testDetectResultRow,
                    matService.getFeatureMat(
                        faceRecognizer,
                        testImgMat,
                        testDetectResultRow
                    )
                );
            }
            knownFaces = parseKnownFaces(
                tempFacesDirPath.toString(),
                faceDetector,
                faceRecognizer
            );
            List<FaceBox> faceBoxList = facePredictorService.predictFaces(
                detectResultFaceFeature,
                knownFaces,
                faceRecognizer
            );
            return imageService.visualizeBoxes(
                testImgPath.toString(),
                faceBoxList
            );
        }
        finally {
            if (faceDetector != null) {
                faceDetector.deallocate();
            }
            if (faceRecognizer != null) {
                faceRecognizer.deallocate();
            }
            for (Mat detectResult: detectResultFaceFeature.keySet()) {
                detectResult.deallocate();
            }
            for (Mat faceFeature: detectResultFaceFeature.values()) {
                faceFeature.deallocate();
            }
            for (List<Mat> matList: knownFaces.values()) {
                for (Mat mat: matList) {
                    if (mat != null) {
                        mat.deallocate();
                    }
                }
            }
            if (tempFacesDirPath != null) {
                fileService.deleteDirRecursive(tempFacesDirPath);
            }
        }
    }

    /**
     * Parses a directory of known faces and returns a map of subject names
     * (denoted by subdirectory names) and their corresponding list of face
     * feature Mats. These Mats are ready to be used for face recognition via
     * {@code FaceRecognizerSF.match()}. <p>
     * 
     * It is the caller's responsibility to properly deallocate the list of
     * Mats in the returned map. <p>
     * 
     * The required structure of the faces directory is as follows. Any file
     * that's not a subdirectory or is not in one, as well as all unsupported
     * image files are ignored.
     * 
     * <pre> <code>
     * known-faces
     * |-- person1
     * |   |-- photo1.png
     * |   `-- photo2.jpg
     * |-- person2
     * |   |--photo1.pgm
     * |   `--photo2.ppm
     * ...
     * </code> </pre>
     * 
     * @param facesDirPath Path of the known faces directory.
     * @param faceDetector YuNet face detection model.
     * @param faceRecognizer SFace face recognition model.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the faces directory is not a
     *                                  directory or does not exist.
     * @throws IOException For general I/O errors.
     * @throws RuntimeException If no valid faces were parsed.
     * @return A map that maps each subject's name to a list of facial feature
     *         Mats associated with them.
     */
    public Map<String, List<Mat>> parseKnownFaces(
        String facesDirPath,
        FaceDetectorYN faceDetector,
        FaceRecognizerSF faceRecognizer
    ) throws IOException {
        File rootDir = new File(facesDirPath);
        if (!rootDir.isDirectory()) {
            throw new IllegalArgumentException(
                "Known faces directory is not a directory or does not exist"
            );
        }
        Map<String, List<Mat>> knownFaces = new HashMap<>();
        Mat imgMat = null, detectResult = null;
        int readImgs = 0;
        try {
            for (File subDir: rootDir.listFiles()) {
                String subDirName = subDir.getName();
                File[] imgFileArr = subDir.listFiles(imgOnlyFilter);
                if (!subDir.isDirectory() || imgFileArr.length == 0) {
                    continue;
                }
                List<Mat> featureMatList = new ArrayList<>();
                knownFaces.put(subDirName, featureMatList);
                for (File imgFile: imgFileArr) {
                    imgMat = imread(imgFile.getAbsolutePath());
                    detectResult =
                        faceDetectorService.detectFaces(imgMat, faceDetector);
                    readImgs += detectResult.rows();
                    featureMatList.addAll(
                        matService.getFeatureMats(
                            faceDetector,
                            faceRecognizer,
                            imgMat,
                            detectResult
                        )
                    );
                }
            }
            if (readImgs == 0) {
                throw new RuntimeException("No valid faces were parsed");
            }
        }
        catch (Exception e) {
            for (List<Mat> matList: knownFaces.values()) {
                for (Mat mat: matList) {
                    if (mat != null) {
                        mat.deallocate();
                    }
                }
            }
            throw e;
        }
        finally {
            if (imgMat != null) {
                imgMat.deallocate();
            }
            if (detectResult != null) {
                detectResult.deallocate();
            }
        }
        return knownFaces;
    }
}
