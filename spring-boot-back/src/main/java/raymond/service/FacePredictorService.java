package raymond.service;

// import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

// import java.io.File;
// import java.io.FileOutputStream;
// import java.io.IOException;
// import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.bytedeco.opencv.opencv_core.Mat;
// import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
// import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import raymond.classes.FaceBox;

/**
 * Utilizes the SFace deep neural netowrk face recognition model. Thank you
 * Professor Weihong Deng, PhD Candidate Zhong Yaoyao, and Master Candidate
 * Chengrui Wang! <p>
 * 
 * https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
 * <p>
 * 
 * Note that not all image formats are supported for facial recognition due to
 * limitations of the {@code cv::imread()} function. You can find a list
 * of supported formats here: <p>
 * 
 * https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
 */
@Service
public class FacePredictorService {

    // @Autowired
    // private FaceDetectorService faceDetectorService;

    // @Autowired
    // private ImageService imageService;

    // @Autowired
    // private MatService matService;

    @Value("${app.service.facerecognizerpath}")
    private String faceRecognizerModelPath;

    /**
     * Creates a new FaceRecognizerSF object.
     * 
     * It is the caller's responsibility to later deallocate the face
     * recognizer properly.
     * 
     * @return A new FaceDetectorSF object.
     */
    public FaceRecognizerSF createFaceRecognizer() {
        FaceRecognizerSF faceRecognizer =
            FaceRecognizerSF.create(faceRecognizerModelPath, "");
        return faceRecognizer;
    }

    /**
     * Identifies faces in the map of face detection result Mats and face
     * feature Mats by comparing them against a set of known faces. Note that
     * not all image formats are supported; see {@link #FacePredictor} for
     * details. <p>
     * 
     * The test faces map and known faces map are not modified or deallocated
     * as a result of the operation; it is the caller's responsibility to later
     * deallocate the two maps.
     * 
     * @param testFaces Map of detection result Mats (typically one row) and
     *                  face feature Mats.
     * @param knownFaceFeatures Map of known subjects and their corresponding
     *                          facial feature Mats.
     * @param faceRecognizer SFace face recognizer model.
     * @return A list of regions of interest (ROIs), each one describing the
     *         coordinates + dimensions of each face as well as the predicted
     *         label and confidence score associated with it.
     * @throws NullPointerException If any arguments are null.
     */
    public List<FaceBox> predictFaces(
        Map<Mat, Mat> testFaces,
        Map<String, List<Mat>> knownFaceFeatures,
        FaceRecognizerSF faceRecognizer
    ) {
        if (testFaces == null) {
            throw new NullPointerException("Test faces map");
        }
        if (knownFaceFeatures == null) {
            throw new NullPointerException("Known faces map");
        }
        if (faceRecognizer == null) {
            throw new NullPointerException("Face recognizer model");
        }
        Map<String, FaceBox> bestMatches = new HashMap<>();
        for (Mat detectResult: testFaces.keySet()) {
            for (String subjName: knownFaceFeatures.keySet()) {
                List<Mat> subjFtMatList = knownFaceFeatures.get(subjName);
                double cosScore = 0.0;
                for (Mat subjFtMat: subjFtMatList) {
                    cosScore += faceRecognizer.match(
                        testFaces.get(detectResult),
                        subjFtMat,
                        FaceRecognizerSF.FR_COSINE
                    );
                }
                double avgCosScore =
                    cosScore / subjFtMatList.size();
                // Mean cosine distance >= 0.363 implies exact match.
                // Otherwise, this face is ignored.
                if (avgCosScore >= 0.363
                        && !bestMatches.containsKey(subjName)
                ) {
                    FaceBox faceBox = new FaceBox(detectResult);
                    faceBox.label = subjName;
                    faceBox.predictScore = avgCosScore;
                    bestMatches.put(subjName, faceBox);
                }
            }
        }
        return new ArrayList<>(bestMatches.values());
    }

    /**
     * Identifies faces in the test image by comparing them against a set of
     * known faces. Note that not all image formats are supported; see
     * {@link #FacePredictor} for details. <p>
     * 
     * The known faces map is not modified or deallocated as a result of the
     * operation; it is the caller's responsibility to later deallocate the
     * list of Mats in the given known faces map.
     * 
     * @param testImgBytes Byte array of the image to predict faces from.
     * @param knownFaces Map of known subjects and their corresponding facial
     *                   feature Mats.
     * @param faceDetector YuNet face detection model.
     * @param faceRecognizer SFace face recognition model.
     * @return A list of regions of interest (ROIs), each one describing the
     *         coordinates + dimensions of each face as well as the predicted
     *         label and confidence score associated with it.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If the image is invalid or for general I/O errors.
     */
    // public List<FaceBox> predictFaces(
    //     byte[] testImgBytes,
    //     Map<String, List<Mat>> knownFaces,
    //     FaceDetectorYN faceDetector,
    //     FaceRecognizerSF faceRecognizer
    // ) throws IOException {
    //     if (testImgBytes == null) {
    //         throw new NullPointerException("Test image byte array");
    //     }
    //     File tempInputFile = File.createTempFile("tempInputImg", null);
    //     try (OutputStream outStream = new FileOutputStream(tempInputFile)) {
    //         outStream.write(testImgBytes);
    //         outStream.flush();
    //         return predictFaces(
    //             tempInputFile.getAbsolutePath(),
    //             knownFaces,
    //             faceDetector,
    //             faceRecognizer
    //         );
    //     }
    //     finally {
    //         tempInputFile.delete();
    //     }
    // }

    /**
     * Identifies faces in the test image by comparing them against a set of
     * known faces. Note that not all image formats are supported; see
     * {@link #FacePredictor} for details. <p>
     * 
     * The known faces map is not modified or deallocated as a result of the
     * operation; it is the caller's responsibility to later deallocate the
     * list of Mats in the given known faces map.
     * 
     * @param testImgPath Path of the test image to predict faces from.
     * @param knownFaces Map of known subjects and their corresponding facial
     *                   feature Mats.
     * @param faceDetector YuNet face detection model.
     * @param faceRecognizer SFace face recognition model.
     * @return A list of regions of interest (ROIs), each one describing the
     *         coordinates + dimensions of each face as well as the predicted
     *         label and confidence score associated with it.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If the image is invalid or for general I/O errors.
     */
    // public List<FaceBox> predictFaces(
    //     String testImgPath,
    //     Map<String, List<Mat>> knownFaceFeatures,
    //     FaceDetectorYN faceDetector,
    //     FaceRecognizerSF faceRecognizer
    // ) throws IOException {
    //     if (testImgPath == null) {
    //         throw new NullPointerException("Test image path");
    //     }
    //     if (knownFaceFeatures == null) {
    //         throw new NullPointerException("Known faces map");
    //     }
    //     if (faceDetector == null) {
    //         throw new NullPointerException("Face detector model");
    //     }
    //     if (faceRecognizer == null) {
    //         throw new NullPointerException("Face recognizer model");
    //     }
    //     List<FaceBox> identifiedFaces = new ArrayList<>();
    //     Mat testImgMat = null, testRoiMat = null, testFeatureMat = null;
    //     try {
    //         testImgMat = imread(testImgPath);
    //         if (testImgMat.data() == null
    //                 || testImgMat.rows() <= 0
    //                 || testImgMat.cols() <= 0
    //         ) {
    //             throw new RuntimeException("Invalid test image");
    //         }
    //         testRoiMat =
    //             faceDetectorService.detectFaces(testImgPath, faceDetector);
    //         for (int i = 0; i < testRoiMat.rows(); ++i) {
    //             try {
    //                 testFeatureMat = matService.getFeatureMat(
    //                     faceRecognizer,
    //                     testImgMat,
    //                     testRoiMat.row(i)
    //                 );
    //                 for (String subjName: knownFaceFeatures.keySet()) {
    //                     List<Mat> subjFeatureMatList = knownFaceFeatures.get(subjName);
    //                     double cosScore = 0;
    //                     for (Mat subjFeatureMat: subjFeatureMatList) {
    //                         cosScore += faceRecognizer.match(
    //                             testFeatureMat,
    //                             subjFeatureMat,
    //                             FaceRecognizerSF.FR_COSINE
    //                         );
    //                     }
    //                     double avgCosScore =
    //                         cosScore / subjFeatureMatList.size();
    //                     // Mean cosine distance >= 0.363 implies exact match.
    //                     // Otherwise, this face is ignored.
    //                     if (avgCosScore >= 0.363) {
    //                         FaceBox roi = new FaceBox(i, testRoiMat);
    //                         roi.label = subjName;
    //                         roi.predictScore = avgCosScore;
    //                         identifiedFaces.add(roi);
    //                     }
    //                 }
    //             }
    //             finally {
    //                 if (testFeatureMat != null) {
    //                     testFeatureMat.deallocate();
    //                     testFeatureMat = null;
    //                 }
    //             }
    //         }
    //     }
    //     finally {
    //         if (testImgMat != null) {
    //             testImgMat.deallocate();
    //         }
    //         if (testRoiMat != null) {
    //             testRoiMat.deallocate();
    //         }
    //     }
    //     return identifiedFaces;
    // }
}
