package raymond.service;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;
import java.io.FileOutputStream;
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
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;
import org.springframework.beans.factory.ObjectFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import raymond.classes.FaceBox;
import raymond.utils.StringUtils;

@Service
public class FaceRecognizerFacade {

    @Autowired
    private FaceDetectorService fdService;

    @Autowired
    private FacePredictorService fpService;

    @Autowired
    private ImageService imgService;

    @Autowired
    private MatService matService;

    @Autowired
    private ObjectFactory<FaceDetectorYN> fdFactory;

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
     * Orchestrates the facial recognition process from start to finish,
     * returning the visualized result as a byte array.
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
        String faceImgFormat =
            StringUtils.getExtensionWithDot(faceImg.getOriginalFilename());
        File tempTestImgFile = File.createTempFile("testImg", testImgFormat);
        File tempFaceImgFile = File.createTempFile("faceImg", faceImgFormat);
        try (FileOutputStream fos = new FileOutputStream(tempTestImgFile)) {
            fos.write(testImg.getBytes());
        }
        try (FileOutputStream fos = new FileOutputStream(tempFaceImgFile)) {
            fos.write(faceImg.getBytes());
        }
        Mat testImgMat = imread(tempTestImgFile.getAbsolutePath());
        Mat faceImgMat = imread(tempFaceImgFile.getAbsolutePath());
        Mat testImgFaceBoxes = null,
            faceImgFaceBoxes = null;
        try (FaceDetectorYN fd = fdFactory.getObject()) {
            testImgFaceBoxes = fdService.detectFaces(testImgMat, fd);
            faceImgFaceBoxes = fdService.detectFaces(faceImgMat, fd).row(0);
        }
        FaceBox predictedFaceBox = fpService.predictFace(
            faceImgMat,
            faceImgFaceBoxes,
            testImgMat,
            testImgFaceBoxes
        );
        return imgService.visualizeBoxes(
            tempTestImgFile.getAbsolutePath(),
            Arrays.asList(predictedFaceBox)
        );
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
        FaceDetectorYN fd
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
                    detectResult = fdService.detectFaces(imgMat, fd);
                    readImgs += detectResult.rows();
                    featureMatList.addAll(
                        matService.createFeatureMats(imgMat, detectResult)
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
                        mat.close();
                    }
                }
            }
            throw e;
        }
        finally {
            if (imgMat != null) {
                imgMat.close();
            }
            if (detectResult != null) {
                detectResult.close();
            }
        }
        return knownFaces;
    }
}
