package raymond;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;

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
public class FacePredictor {

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
     * known faces. <br></br>
     * 
     * Note that not all image formats are supported; see {@link #FacePredictor}
     * for details. <br></br>
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
     * `-- person2
     * |   |--photo1.pgm
     * |   `--photo2.ppm
     * ...
     * </code> </pre>
     * 
     * @param testImgPath Path of the test image to predict faces from.
     * @param facesDirPath Path of known faces to compare against.
     * @param detectorModelPath Path of the YuNet face detection model.
     * @param recognizerModelPath Path of the SFace face recognition model.
     * @return A list of regions of interest (ROIs), each one describing the
     *         coordinates + dimensions of each face as well as the predicted
     *         label and confidence score associated with it.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If the image is invalid or for general I/O errors.
     */
    public static List<FrameData> predictFaces(
        String testImgPath,
        String facesDirPath,
        String detectorModelPath,
        String recognizerModelPath
    ) throws IOException {
        if (testImgPath == null) {
            throw new NullPointerException("Test image path");
        }
        if (facesDirPath == null) {
            throw new NullPointerException("Known faces directory path");
        }
        if (detectorModelPath == null) {
            throw new NullPointerException("Detector model path");
        }
        if (recognizerModelPath == null) {
            throw new NullPointerException("Recognizer model path");
        }
        Mat testImgMat = imread(testImgPath);
        if (testImgMat.data() == null
                || testImgMat.rows() <= 0
                || testImgMat.cols() <= 0
        ) {
            throw new RuntimeException("Invalid test image");
        }
        FaceRecognizerSF faceRecognizer =
            FaceRecognizerSF.create(recognizerModelPath, "");
        Map<String, List<Mat>> knownFaces = parseKnownFaces(
            facesDirPath,
            detectorModelPath,
            recognizerModelPath
        );
        Mat testRoiMat =
            FaceDetector.detectFaces(testImgPath,detectorModelPath);
        List<FrameData> roiList = new ArrayList<>();
        for (int i = 0; i < testRoiMat.rows(); ++i) {
            System.out.println("Subject " + i);
            Mat testFeatureMat = ImageUtils.getFeatureMat(
                faceRecognizer,
                testImgMat,
                testRoiMat.row(i)
            );
            String predictedSubj = "";
            double minCosScore = Double.MIN_VALUE;
            for (String subjName: knownFaces.keySet()) {
                System.out.println("Comparing against " + subjName);
                List<Mat> subjFeatureMatList = knownFaces.get(subjName);
                double cosScore = 0;
                for (Mat subjFeatureMat: subjFeatureMatList) {
                    cosScore += faceRecognizer.match(
                        testFeatureMat,
                        subjFeatureMat,
                        FaceRecognizerSF.FR_COSINE
                    );
                }
                // Mean cosine distance >= 0.363 implies exact identity match.
                // If no exact match found, we identify based on closest match.
                double avgCosScore = cosScore / subjFeatureMatList.size();
                if (avgCosScore > minCosScore) {
                    predictedSubj = subjName;
                    minCosScore = avgCosScore;
                }
                System.out.println("Average score " + avgCosScore);
            }
            FrameData roi = new FrameData(i, testRoiMat);
            roi.label = predictedSubj;
            roi.predictScore = minCosScore;
            roiList.add(roi);
        }
        testImgMat.deallocate();
        return roiList;
    }

    /**
     * Helper function that parses a directory of known faces and returns a map
     * of subject names (denoted by subdirectory names) and their corresponding
     * collection of facial feature Mats. <br></br>
     * 
     * These Mats are ready to be used for face recognition via <code>
     * FaceRecognizerSF.match()</code>. Not-yet recognized faces in test images
     * will be compared against these known facial features in order to
     * identify them.
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
    private static Map<String, List<Mat>> parseKnownFaces(
        String facesDirPath,
        String detectorModelPath,
        String recognizerModelPath
    ) throws IOException {
        File rootDir = new File(facesDirPath);
        if (!rootDir.isDirectory()) {
            throw new IllegalArgumentException(
                "Known faces directory is not a directory or does not exist"
            );
        }
        Map<String, List<Mat>> knownFaces = new HashMap<>();
        FaceRecognizerSF faceRecognizer =
            FaceRecognizerSF.create(recognizerModelPath, "");
        int readImgs = 0;
        for (File subDir: rootDir.listFiles()) {
            String subDirName = subDir.getName();
            File[] imgFileArr = subDir.listFiles(imgOnlyFilter);
            if (!subDir.isDirectory() || imgFileArr.length == 0) {
                continue;
            }
            List<Mat> subjectFaces = new ArrayList<>();
            knownFaces.put(subDirName, subjectFaces);
            for (File imgFile: imgFileArr) {
                Mat imgMat = imread(imgFile.getAbsolutePath());
                Mat roiMat = FaceDetector.detectFaces(
                    imgMat,
                    detectorModelPath
                );
                // There should ideally only be one ROI per image. But if
                // multiple instances of the same subject appear in one image,
                // the program will accomodate for this case.
                for (int i = 0; i < roiMat.rows(); ++i) {
                    Mat featureMat = ImageUtils.getFeatureMat(
                        faceRecognizer,
                        imgMat,
                        roiMat.row(i)
                    );
                    subjectFaces.add(featureMat);
                    ++readImgs;
                }
                roiMat.deallocate();
                imgMat.deallocate();
            }
        }
        faceRecognizer.deallocate();
        if (readImgs == 0) {
            throw new RuntimeException("No valid faces were parsed");
        }
        return knownFaces;
    }
}
