package raymond;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.HashSet;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer;

/**
 * Note that not all image formats are supported for model training due to
 * limitations of the <code>cv::imread()</code> function. You can find a list
 * of supported formats here: <br></br>
 * 
 * https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
 */
public class FaceRecognizerTrainer {
    
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
        FaceRecognizer faceRecognizer = createTrainModel(imgList, labelList);
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
     * starting at the integer 0 and ascending. <br></br>
     * 
     * The two lists will be cleared before populating and will end up parallel
     * such that the image represented by <code>imgList.get(k)</code>
     * corresponds to the label represented by <code>labelList.get(k)</code>.
     * <br></br>
     * 
     * Only images with a supported file extension (which should indicate its
     * format) will be parsed. See the class Javadoc for details on supported
     * image formats. <br></br>
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
     * @see #CreateFaceRecognizer
     * 
     * @param trainingPath Path of the directory to read images from.
     * @param imgList List to store image files in, typically empty.
     * @param labelList List to store corresponding lbales in, typically empty.
     * @throws NullPointerException If any arguments are null.
     * @throws IllegalArgumentException If the training path is invalid or
     *                                  cannot be accessed.
     * @throws RuntimeException If no valid images are found.
     */
    public static void parseTrainingFaces(
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
        int label = 0, readImgs = 0;
        for (File subDir: rootDir.listFiles()) {
            if (!subDir.isDirectory() || subDir.listFiles().length == 0) {
                continue;
            }
            for (File img: subDir.listFiles(imgOnlyFilter)) {
                imgList.add(img);
                labelList.add(label);
                ++readImgs;
            }
            ++label;
        }
        if (readImgs == 0) {
            throw new RuntimeException("No valid training images to parse");
        }
    }

    /**
     * Creates, trains, and returns an OpenCV FaceRecognizer using the provided
     * image list and label list. The resulting FaceRecognizer will have some
     * capacity to identify faces in the provided images in subsequent images.
     * <br></br>
     * 
     * Note that not all image formats are supported and invalid files or
     * image formats will be skipped over with a warning from the console. See
     * the class Javadoc for more info.
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
        FaceRecognizer faceRecognizer = LBPHFaceRecognizer.create();
        int totalImgs = 0;
        int batchSize = 50;
        // Feeds the model batches of samples of size defined above to prevent
        // a C++ stack overflow.
        for (int i = 0; i < imgList.size(); i += batchSize) {
            int endIdx = Math.min(i + batchSize, imgList.size());
            List<File> imgBatch = imgList.subList(i, endIdx);
            List<Mat> imgMatList = new ArrayList<>();
            for (int j = 0; j < imgBatch.size(); ++j) {
                File imgFile = imgBatch.get(j);
                String imgFileAbsPath = imgFile.getAbsolutePath();
                Mat imgMat = imread(
                    imgFileAbsPath,
                    IMREAD_GRAYSCALE
                );
                if (imgMat.data() == null ||
                        imgMat.rows() <= 0 ||
                        imgMat.cols() <= 0
                ) {
                    System.out.println(
                        "Invalid/unsupported file: " + imgFileAbsPath
                    );
                    imgMat.deallocate();
                    continue;
                }
                System.out.println("Processing: " + imgFileAbsPath);
                imgMatList.add(imgMat);
                ++totalImgs;
            }
            if (imgMatList.size() == 0) {
                continue;
            }
            // Seems like the size of the image MatVector needs to exactly
            // match the number of Mats inserted, otherwise C++ throws this:
            // error: (-215:Assertion failed) s >= 0 in function 'setSize'
            MatVector imgVec = new MatVector(imgMatList.size());
            Mat labelsMat = new Mat(imgMatList.size(), 1, CV_32SC1);
            IntBuffer labelBuf = labelsMat.createBuffer();
            for (int k = 0; k < imgMatList.size(); ++k) {
                imgVec.put(k, imgMatList.get(k));
                labelBuf.put(k, labelList.get(k));
            }
            System.out.println(
                "Training model using " + imgVec.size() + " samples"
            );
            // Don't discard previous data learned by using train()
            faceRecognizer.update(imgVec, labelsMat);
            imgVec.deallocate();
            labelsMat.deallocate();
        }
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
