package raymond;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;

/**
 * Note that not all image formats are supported for model training due to
 * limitations of the <code>cv::imread()</code> function. You can find a list
 * of supported formats here: <br></br>
 * 
 * https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
 */
public class FaceTrainer {
    
    public static void main(String[] args) {
        // if (args.length < 2) {
        //     System.out.println(
        //         "Usage: [program] [training-dir] [save-model-dir]"
        //     );
        //     System.exit(0);
        // }
        // File trainingDir = new File(args[0]);
        // if (!trainingDir.exists() || !trainingDir.isDirectory()) {
        //     System.out.println(
        //         "Invalid training directory " + args[0]
        //     );
        //     System.exit(0);
        // }
        // File saveModelDir = new File(args[1]);
        // if (saveModelDir.getName().indexOf('.') != -1) {
        //     System.out.println(
        //         "Invalid save directory " + args[1]
        //     );
        // }
        // if (!saveModelDir.exists()) {
        //     saveModelDir.mkdir();
        // }
        // if (!saveModelDir.isDirectory()) {
        //     System.out.println(
        //         "Invalid save directory " + args[1]
        //     );
        // }
        final File trainingDir = new File("training-faces");
        final File saveModelDir = new File("models");

        List<File> imgList = new ArrayList<>();
        List<Integer> labelList = new ArrayList<>();
        parseTrainingFaces(trainingDir.getAbsolutePath(), imgList, labelList);
        FaceRecognizer faceRecognizer = createTrainModel(imgList, labelList, true);
        File saveModelFile = new File(
            saveModelDir.getAbsolutePath() + "/face-recognizer.xml"
        );
        saveModelFile.delete();
        faceRecognizer.write(saveModelFile.getAbsolutePath());
    }

    /**
     * Populates a list of image files and a list of labels by reading a
     * directory of training face images. Each subdirectory represents a
     * different subject, and labels are automatically assigned to each subject
     * starting at the integer 0 and ascending. Returns a map of parsed labels
     * and their corresponding subject names. <br></br>
     * 
     * At least two subjects are required to perform a linear dimension
     * analysis (as required by the Fisherfaces and Eigenfaces models).
     * <br></br>
     * 
     * The two lists will be cleared before populating and will end up parallel
     * such that the image represented by <code>imgList.get(k)</code>
     * corresponds to the label represented by <code>labelList.get(k)</code>.
     * <br></br>
     * 
     * Only images with a supported file extension (which should indicate its
     * format) will be parsed. See {@link #FaceTrainer} for details. <br></br>
     * 
     * Required training directory structure:
     * 
     * <pre> <code>
     * training-faces
     * |-- person1
     * |   |-- photo1.png
     * |   `-- photo2.jpg
     * `-- person2
     * |   |--photo1.pgm
     * |   `--photo2.ppm
     * ...
     * </code> </pre>
     * 
     * @param trainingPath Path of the directory to read images from.
     * @param imgList List to store image files in, typically empty.
     * @param labelList List to store corresponding lbales in, typically empty.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the training path is invalid or
     *                                  cannot be accessed.
     * @throws RuntimeException If no valid images are found or if less than
     *                          two subjects are parsed.
     */
    public static Map<Integer, String> parseTrainingFaces(
        String trainingPath,
        List<File> imgList,
        List<Integer> labelList
    ) {
        if (trainingPath == null) {
            throw new NullPointerException(
                "Training path is null"
            );
        }
        if (imgList == null) {
            throw new NullPointerException(
                "Image list is null"
            );
        }
        if (labelList == null) {
            throw new NullPointerException(
                "Label list is null"
            );
        }
        File rootDir = new File(trainingPath);
        if (!rootDir.exists() || !rootDir.isDirectory()) {
            throw new IllegalArgumentException(
                "Training directory is not a directory or does not exist"
            );
        }
        imgList.clear();
        labelList.clear();
        // See method Javadoc for info about accepted formats
        Set<String> acceptedFormats = new HashSet<String>(
            Arrays.asList(
                "bmp", "dib", "jpeg", "jpg", "jpe", "jp2", "png", "webp",
                "avif", "pbm", "pgm", "ppm", "pxm", "pnm", "pfm", "sr", "ras",
                "tiff", "tif", "exr", "hdr", "pic"
            )
        );
        FilenameFilter imgOnlyFilter = new FilenameFilter() {
            public boolean accept(File file, String name) {
                String nameLower = name.toLowerCase();
                int extSepIdx = nameLower.lastIndexOf('.');
                if (extSepIdx == -1) {
                    return false;
                }
                String fileSuffix = nameLower.substring(extSepIdx + 1);
                return acceptedFormats.contains(fileSuffix);
            }
        };
        Map<Integer, String> labelLegend = new HashMap<>();
        int label = 0, readImgs = 0;
        for (File subDir: rootDir.listFiles()) {
            String subDirName = subDir.getName();
            File[] imgFileList = subDir.listFiles(imgOnlyFilter);
            if (!subDir.isDirectory() || imgFileList.length == 0) {
                continue;
            }
            labelLegend.put(label, subDirName);
            for (File img: imgFileList) {
                imgList.add(img);
                labelList.add(label);
                ++readImgs;
            }
            ++label;
        }
        if (readImgs == 0) {
            throw new RuntimeException("No valid training images to parse");
        }
        if (labelLegend.size() < 2) {
            throw new RuntimeException(
                "At least two subjects are required to perform a linear " +
                "dimension analysis (LDA)"
            );
        }
        System.out.println(
            "Successfully parsed " + readImgs + " training samples"
        );
        for (int parsedLabel: labelLegend.keySet()) {
            System.out.println(
                String.format(
                    "Label %d: %s",
                    parsedLabel,
                    labelLegend.get(parsedLabel)
                )
            );
        }
        return labelLegend;
    }

    /**
     * Creates, trains, and returns an OpenCV FaceRecognizer using the provided
     * image list and label list. The resulting FaceRecognizer will have some
     * capacity to identify faces in the provided images in subsequent images.
     * <br></br>
     * 
     * Note that not all image formats are supported and invalid files or
     * image formats will be skipped over with a warning from the console. See
     * {@link #FaceTrainer} for details.
     * 
     * @param imgList List of training image files.
     * @param labelList List of corresponding training image labels.
     * @return A FaceRecognizer trained to recognize the provided faces.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the image and label lists are
     *                                  different in size.
     * @throws RuntimeException If no valid images are passed in.
     */
    public static FaceRecognizer createTrainModel(
        List<File> imgList,
        List<Integer> labelList
    ) {
        return createTrainModel(imgList, labelList, false);
    }

    /**
     * Underlying method with an extra parameter to enable or disable debugging
     * mode. <br></br>
     * 
     * If <code>debug == true</code>, all grayscale sqaure images converted
     * from training images will be written to a debug directory where they can
     * be manually inspected. These are the images that the FaceRecognizer
     * model is trained with. If a debug directory already exists, it will be
     * deleted and remade before new images are written. <br></br>
     * 
     * See {@link #createTrainModel(List, List)}.
     * 
     * @param imgList List of training image files.
     * @param labelList List of corresponding training image labels.
     * @param debug Enable or disable debug mode.
     * @return A FaceRecognizer trained to recognize the provided faces.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the image and label lists are
     *                                  different in size.
     * @throws RuntimeException If no valid images are passed in.
     */
    public static FaceRecognizer createTrainModel(
        List<File> imgList,
        List<Integer> labelList,
        boolean debug
    ) {
        if (imgList == null) {
            throw new NullPointerException(
                "Image list is null or empty"
            );
        }
        if (labelList == null) {
            throw new NullPointerException(
                "Label list is null or empty"
            );
        }
        if (imgList.size() != labelList.size()) {
            throw new IllegalArgumentException(
                "Image and label lists not equal in size"
            );
        }
        File debugImgDir = null;
        if (debug) {
            debugImgDir = new File("./training-debug");
            debugImgDir.delete();
            debugImgDir.mkdir();
        }
        FaceRecognizer faceRecognizer = FisherFaceRecognizer.create();
        int totalImgs = 0;
        List<Mat> imgMatList = new ArrayList<>();
        for (int i = 0; i < imgList.size(); ++i) {
            File imgFile = imgList.get(i);
            String imgFileAbsPath = imgFile.getAbsolutePath();
            Mat grayscaleImgMat = imread(
                imgFileAbsPath,
                IMREAD_GRAYSCALE
            );
            Mat resizedImgMat = ImageUtils.squareMat(grayscaleImgMat, 256);
            grayscaleImgMat.deallocate();
            if (resizedImgMat.data() == null ||
                    resizedImgMat.rows() <= 0 ||
                    resizedImgMat.cols() <= 0
            ) {
                System.out.println(
                    "Invalid/unsupported file: " + imgFileAbsPath
                );
                resizedImgMat.deallocate();
                continue;
            }
            System.out.println("Training: " + imgFileAbsPath);
            // Write gray square images to files for manual inspection
            if (debug) {
                String debugImgPath = String.format(
                    "%s/%s",
                    debugImgDir.getAbsolutePath(),
                    imgFile.getName()
                );
                imwrite(debugImgPath, resizedImgMat);
                System.out.println("Saved debug image: " + debugImgPath);
            }
            imgMatList.add(resizedImgMat);
            ++totalImgs;
        }
        // Seems like the size of the image MatVector needs to exactly match
        // the number of Mats inserted, otherwise an assertion error is thrown.
        MatVector imgVec = new MatVector(imgMatList.size());
        Mat labelsMat = new Mat(imgMatList.size(), 1, CV_32SC1);
        IntBuffer labelBuf = labelsMat.createBuffer();
        for (int i = 0; i < imgMatList.size(); ++i) {
            imgVec.put(i, imgMatList.get(i));
            labelBuf.put(i, labelList.get(i));
        }
        faceRecognizer.train(imgVec, labelsMat);
        // Free pointers to all C++ Mats
        for (int i = 0; i < imgVec.size(); ++i) {
            imgVec.get(i).deallocate();
        }
        imgVec.deallocate();
        labelsMat.deallocate();
        if (totalImgs == 0) {
            throw new RuntimeException(
                "No valid training images to feed model with"
            );
        }
        System.out.println(
            "Successfully trained model using " + totalImgs + " total samples"
        );
        return faceRecognizer;
    }
}
