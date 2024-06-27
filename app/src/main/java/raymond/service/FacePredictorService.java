package raymond.service;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import raymond.classes.ROIData;
import raymond.utils.StringUtils;

/**
 * Utilizes the SFace deep neural netowrk face recognition model. Thank you
 * Professor Deng, Ph.D Candidate Zhong, and Master Candidate Wang! <br></br>
 * 
 * https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
 * <br></br>
 * 
 * Note that not all image formats are supported for facial recognition due to
 * limitations of the <code>cv::imread()</code> function. You can find a list
 * of supported formats here: <br></br>
 * 
 * https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
 */
@Service
public class FacePredictorService {

    @Autowired
    public ImageService imageService;

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
     * Identifies faces in the test image by comparing them against a set of
     * known faces. Note that not all image formats are supported; see
     * {@link #FacePredictor} for details. <br></br>
     * 
     * The known faces map is not modified or deallocated as a result of the
     * operation; it is the caller's responsibility to later deallocate the
     * list of Mats in the given known faces map.
     * 
     * @param testImgBytes Byte array of the image to predict faces from.
     * @param knownFaces Map of known subjects and their corresponding facial
     *                   feature Mats.
     * @param detectorModelPath Path of the YuNet face detection model.
     * @param faceRecognizer SFace face recognition model.
     * @return A list of regions of interest (ROIs), each one describing the
     *         coordinates + dimensions of each face as well as the predicted
     *         label and confidence score associated with it.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If the image is invalid or for general I/O errors.
     */
    public List<ROIData> predictFaces(
        byte[] testImgBytes,
        Map<String, List<Mat>> knownFaces,
        String detectorModelPath,
        FaceRecognizerSF faceRecognizer
    ) throws IOException {
        if (testImgBytes == null) {
            throw new NullPointerException("Test image byte array");
        }
        File tempInputFile = File.createTempFile("tempInputImg", null);
        try (OutputStream outStream = new FileOutputStream(tempInputFile)) {
            outStream.write(testImgBytes);
            outStream.flush();
            return predictFaces(
                tempInputFile.getAbsolutePath(),
                knownFaces,
                detectorModelPath,
                faceRecognizer
            );
        }
        finally {
            tempInputFile.delete();
        }
    }

    /**
     * Identifies faces in the test image by comparing them against a set of
     * known faces. Note that not all image formats are supported; see
     * {@link #FacePredictor} for details. <br></br>
     * 
     * <strong> It is the caller's responsibility to later deallocate the list
     * of Mats in the given known faces map. </strong>
     * 
     * @param testImgPath Path of the test image to predict faces from.
     * @param knownFaces Map of known subjects and their corresponding facial
     *                   feature Mats.
     * @param detectorModelPath Path of the YuNet face detection model.
     * @param faceRecognizer SFace face recognition model.
     * @return A list of regions of interest (ROIs), each one describing the
     *         coordinates + dimensions of each face as well as the predicted
     *         label and confidence score associated with it.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If the image is invalid or for general I/O errors.
     */
    public List<ROIData> predictFaces(
        String testImgPath,
        Map<String, List<Mat>> knownFaces,
        String detectorModelPath,
        FaceRecognizerSF faceRecognizer
    ) throws IOException {
        if (testImgPath == null) {
            throw new NullPointerException("Test image path");
        }
        if (knownFaces == null) {
            throw new NullPointerException("Known faces map");
        }
        if (detectorModelPath == null) {
            throw new NullPointerException("Face detector model path");
        }
        if (faceRecognizer == null) {
            throw new NullPointerException("Face recognizer model");
        }
        List<ROIData> identifiedFaces = new ArrayList<>();
        Mat testImgMat = null, testRoiMat = null, testFeatureMat = null;
        try {
            testImgMat = imread(testImgPath);
            if (testImgMat.data() == null
                    || testImgMat.rows() <= 0
                    || testImgMat.cols() <= 0
            ) {
                throw new RuntimeException("Invalid test image");
            }
            testRoiMat =
                FaceDetectorService.detectFaces(testImgPath, detectorModelPath);
            for (int i = 0; i < testRoiMat.rows(); ++i) {
                try {
                    testFeatureMat = imageService.getFeatureMat(
                        faceRecognizer,
                        testImgMat,
                        testRoiMat.row(i)
                    );
                    for (String subjName: knownFaces.keySet()) {
                        List<Mat> subjFeatureMatList = knownFaces.get(subjName);
                        double cosScore = 0;
                        for (Mat subjFeatureMat: subjFeatureMatList) {
                            cosScore += faceRecognizer.match(
                                testFeatureMat,
                                subjFeatureMat,
                                FaceRecognizerSF.FR_COSINE
                            );
                        }
                        double avgCosScore =
                            cosScore / subjFeatureMatList.size();
                        // Mean cosine distance >= 0.363 implies exact match
                        if (avgCosScore >= 0.363) {
                            ROIData roi = new ROIData(i, testRoiMat);
                            roi.label = subjName;
                            roi.predictScore = avgCosScore;
                            identifiedFaces.add(roi);
                        }
                    }
                }
                finally {
                    if (testFeatureMat != null) {
                        testFeatureMat.deallocate();
                        testFeatureMat = null;
                    }
                }
            }
        }
        finally {
            if (testImgMat != null) {
                testImgMat.deallocate();
            }
            if (testRoiMat != null) {
                testRoiMat.deallocate();
            }
        }
        return identifiedFaces;
    }

    /**
     * Parses a directory of known faces and returns a map of subject names
     * (denoted by subdirectory names) and their corresponding collection of
     * facial feature Mats. These Mats are ready to be used for face
     * recognition via <code>FaceRecognizerSF.match()</code>. Not-yet
     * recognized faces in test images will be compared against these known
     * facial features in order to identify them. <br></br>
     * 
     * It is the caller's responsibility to properly deallocate the list of
     * Mats in the returned map. <br></br>
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
     * @param detectorModelPath Path of the YuNet face detection model.
     * @param recognizerModelPath Path of the SFace face recognition model.
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
        String detectorModelPath,
        String recognizerModelPath
    ) throws IOException {
        try (FaceRecognizerSF faceRecognizer = 
                FaceRecognizerSF.create(recognizerModelPath, "")
        ) {
            return parseKnownFaces(
                facesDirPath,
                detectorModelPath,
                faceRecognizer
            );
        }
    }

    /**
     * Parses a directory of known faces and returns a map of subject names
     * (denoted by subdirectory names) and their corresponding collection of
     * facial feature Mats. These Mats are ready to be used for face
     * recognition via <code>FaceRecognizerSF.match()</code>. Not-yet
     * recognized faces in test images will be compared against these known
     * facial features in order to identify them. <br></br>
     * 
     * It is the caller's responsibility to properly deallocate the list of
     * Mats in the returned map. <br></br>
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
     * @param detectorModelPath Path of the YuNet face detection model.
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
        String detectorModelPath,
        FaceRecognizerSF faceRecognizer
    ) throws IOException {
        File rootDir = new File(facesDirPath);
        if (!rootDir.isDirectory()) {
            throw new IllegalArgumentException(
                "Known faces directory is not a directory or does not exist"
            );
        }
        Map<String, List<Mat>> knownFaces = new HashMap<>();
        int readImgs = 0;
        Mat imgMat = null, roiMat = null, featureMat = null;
        try {
            for (File subDir: rootDir.listFiles()) {
                String subDirName = subDir.getName();
                File[] imgFileArr = subDir.listFiles(imgOnlyFilter);
                if (!subDir.isDirectory() || imgFileArr.length == 0) {
                    continue;
                }
                List<Mat> subjectFaces = new ArrayList<>();
                knownFaces.put(subDirName, subjectFaces);
                for (File imgFile: imgFileArr) {
                    try {
                        imgMat = imread(imgFile.getAbsolutePath());
                        roiMat = FaceDetectorService.detectFaces(
                            imgMat,
                            detectorModelPath
                        );
                        // Ideally one person per image, but multiple instances
                        // of the same person will still work.
                        for (int i = 0; i < roiMat.rows(); ++i) {
                            featureMat = imageService.getFeatureMat(
                                faceRecognizer,
                                imgMat,
                                roiMat.row(i)
                            );
                            subjectFaces.add(featureMat);
                            ++readImgs;
                        }
                    }
                    finally {
                        if (imgMat != null) {
                            imgMat.deallocate();
                            imgMat = null;
                        }
                        if (roiMat != null) {
                            roiMat.deallocate();
                            roiMat = null;
                        }
                    }
                }
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
        if (readImgs == 0) {
            throw new RuntimeException("No valid faces were parsed");
        }
        return knownFaces;
    }
}
