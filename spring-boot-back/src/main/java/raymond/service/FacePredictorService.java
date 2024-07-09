package raymond.service;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.springframework.beans.factory.annotation.Autowired;
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

    @Autowired
    private MatService matService;

    @Autowired
    private FaceRecognizerSF fr;

    /**
     * Identifies the target face within the given test image. If the target
     * face cannot be identified in the test image, the method returns null.
     * Note that not all image formats are supported; see
     * {@link #FacePredictor} for details. <p>
     * 
     * The arguments are not modified or deallocated as a result of the
     * operation; it is the caller's responsibility to later deallocate them.
     * 
     * @param targetFaceImg Source image Mat of the target face.
     * @param targetFaceBox Face box of the target face determined using the
     *                      YuNet face detection model.
     * @param testImg Source image Mat of the test image.
     * @param testFaceBoxes Face boxes of the target face determined using the
     *                      YuNet face detection model.
     * @return A FaceBox representing the best match face from the test image,
     *         if any matches exist.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If any Mats are invalid.
     */
    public FaceBox predictFace(
        Mat targetFaceImg,
        Mat targetFaceBox,
        Mat testImg,
        Mat testFaceBoxes
    ) {
        Objects.requireNonNull(targetFaceImg, "Target face Mat");
        Objects.requireNonNull(targetFaceBox, "Target image face boxes");
        Objects.requireNonNull(testImg, "Test image Mat");
        Objects.requireNonNull(testFaceBoxes, "Test image face boxes");
        Objects.requireNonNull(fr, "Face recognizer model");
        if (targetFaceImg.empty()) {
            throw new IllegalArgumentException("Invalid target face image");
        }
        if (targetFaceBox.empty()) {
            throw new IllegalArgumentException("Invalid target face boxes");
        }
        if (testImg.empty()) {
            throw new IllegalArgumentException("Invalid test image");
        }
        if (testFaceBoxes.empty()) {
            throw new IllegalArgumentException("Invalid test face boxes");
        }
        Mat targetFeature = null;
        try {
            targetFeature = matService.createFeatureMat(
                targetFaceImg,
                targetFaceBox
            );
            double maxScore = Double.MIN_VALUE;
            Mat testFeature = null, predictedFaceBox = null;
            for (int i = 0; i < testFaceBoxes.rows(); ++i) {
                double cosScore = Double.MIN_VALUE;
                try {
                    testFeature = matService.createFeatureMat(
                        testImg,
                        testFaceBoxes.row(i)
                    );
                    cosScore = fr.match(
                        targetFeature,
                        testFeature,
                        FaceRecognizerSF.FR_COSINE
                    );
                }
                finally {
                    if (testFeature != null) {
                        testFeature.close();
                    }
                }
                // Mean cosine distance >= 0.363 implies exact match.
                // Otherwise, this face is ignored.
                if (cosScore > 0.363 && cosScore > maxScore) {
                    maxScore = cosScore;
                    predictedFaceBox = testFaceBoxes.row(i);
                }
            }
            if (predictedFaceBox == null) {
                return null;
            }
            FaceBox faceBox = new FaceBox(predictedFaceBox);
            faceBox.label = "you";
            faceBox.predictScore = maxScore;
            return faceBox;
        }
        finally {
            if (targetFeature != null) {
                targetFeature.close();
            }
        }
    }

    /**
     * Identifies faces in the map of face boxes and corresponding face
     * feature Mats by comparing them against a set of known faces. Note that
     * not all image formats are supported; see {@link #FacePredictor} for
     * details. <p>
     * 
     * The arguments are not modified or deallocated as a result of the
     * operation; it is the caller's responsibility to later deallocate them.
     * 
     * @param testImg Source image Mat.
     * @param testImgFaceBoxes Face detection result Mat.
     * @param knownFaceFeatures Map of known subjects and their corresponding
     *                          facial feature Mats.
     * @return A list of regions of interest (ROIs), each one describing the
     *         coordinates + dimensions of each face as well as the predicted
     *         label and confidence score associated with it.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the test image or face boxes Mat are
     *                                  invalid.
     */
    public List<FaceBox> predictFaces(
        Mat testImg,
        Mat testImgFaceBoxes,
        Map<String, List<Mat>> knownFaceFeatures
    ) {
        Objects.requireNonNull(testImg, "Source image Mat");
        Objects.requireNonNull(testImgFaceBoxes, "Detect results Mat");
        Objects.requireNonNull(knownFaceFeatures, "Known face features map");
        Objects.requireNonNull(fr, "Face recognizer model");
        if (testImg.empty()) {
            throw new IllegalArgumentException("Invalid test image");
        }
        if (testImgFaceBoxes.empty()) {
            throw new IllegalArgumentException("Invalid test image face boxes");
        }
        Map<Mat, Mat> faceBoxToFaceFeature = new HashMap<>();
        try {
            for (int i = 0; i < testImgFaceBoxes.rows(); ++i) {
                faceBoxToFaceFeature.put(
                    testImgFaceBoxes.row(i),
                    matService.createFeatureMat(
                        testImg,
                        testImgFaceBoxes.row(i)
                    )
                );
            }
            return predictFaces(
                faceBoxToFaceFeature,
                knownFaceFeatures
            );
        }
        finally {
            for (Mat detectResult: faceBoxToFaceFeature.keySet()) {
                detectResult.close();
            }
            for (Mat featureMat: faceBoxToFaceFeature.values()) {
                featureMat.close();
            }
        }
    }

    /**
     * Identifies faces in the map of face detection result Mats and face
     * feature Mats by comparing them against a set of known faces. Note that
     * not all image formats are supported; see {@link #FacePredictor} for
     * details. <p>
     * 
     * The arguments are not modified or deallocated as a result of the
     * operation; it is the caller's responsibility to later deallocate them.
     * 
     * @param testFaces Map of face box Mats and their corresponding face
     *                  feature Mats.
     * @param knownFaceFeatures Map of known subjects and their corresponding
     *                          facial feature Mats.
     * @return A list of regions of interest (ROIs), each one describing the
     *         coordinates + dimensions of each face as well as the predicted
     *         label and confidence score associated with it.
     * @throws NullPointerException If any arguments are null.
     */
    public List<FaceBox> predictFaces(
        Map<Mat, Mat> testFaces,
        Map<String, List<Mat>> knownFaceFeatures
    ) {
        Objects.requireNonNull(testFaces, "Test faces map");
        Objects.requireNonNull(knownFaceFeatures, "Known face features map");
        Objects.requireNonNull(fr, "Face recognizer model");
        Map<String, FaceBox> bestMatches = new HashMap<>();
        for (Mat detectResult: testFaces.keySet()) {
            for (String subjName: knownFaceFeatures.keySet()) {
                List<Mat> subjFtMatList = knownFaceFeatures.get(subjName);
                double cosScore = 0.0;
                for (Mat subjFtMat: subjFtMatList) {
                    cosScore += fr.match(
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
}
