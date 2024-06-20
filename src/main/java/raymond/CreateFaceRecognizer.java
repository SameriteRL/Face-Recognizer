package raymond;

import static org.bytedeco.opencv.global.opencv_core.CV_32SC1;
import static org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.io.File;
import java.nio.IntBuffer;
import java.util.ArrayList;
import java.util.List;

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
public class CreateFaceRecognizer {
    
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
        faceRecognizer.write(
            saveModelDir.getAbsolutePath() + "/face-recognizer.xml"
        );
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
     * Note that all files in the training subdirectories will be parsed
     * indiscriminately, regardless of whether it is a valid image or not.
     * <br></br>
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
     * @throws RuntimeException If no valid images are found.
     */
    private static void parseTrainingFaces(
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
        int label = 0, readImgs = 0;
        for (File subDir: rootDir.listFiles()) {
            if (!subDir.isDirectory() || subDir.listFiles().length == 0) {
                continue;
            }
            for (File img: subDir.listFiles()) {
                imgList.add(img);
                labelList.add(label);
                ++readImgs;
            }
            ++label;
        }
        if (readImgs == 0) {
            throw new RuntimeException("No training images found");
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
     */
    private static FaceRecognizer createTrainModel(
        List<File> imgList,
        List<Integer> labelList
    ) {
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
                Mat imgMat = imread(
                    imgFile.getAbsolutePath(),
                    IMREAD_GRAYSCALE
                );
                if (imgMat.data() == null ||
                        imgMat.rows() <= 0 ||
                        imgMat.cols() <= 0
                ) {
                    System.out.println(
                        "Invalid/unsupported file: "
                        + imgFile.getAbsolutePath()
                    );
                    imgMat.deallocate();
                    continue;
                }
                imgMatList.add(imgMat);
                ++totalImgs;
            }
            // Seems like the size of the image MatVector needs to exactly
            // match the number of Mats inserted, otherwise C++ throws this:
            // error: (-215:Assertion failed) s >= 0 in function 'setSize'
            MatVector imgVec = new MatVector(imgMatList.size());
            Mat labelsMat = new Mat(imgMatList.size(), 1, CV_32SC1);
            IntBuffer labelBuf = labelsMat.createBuffer();
            for (int j = 0; j < imgMatList.size(); ++j) {
                imgVec.put(j, imgMatList.get(j));
                labelBuf.put(j, j);
            }
            faceRecognizer.train(imgVec, labelsMat);
            imgVec.deallocate();
            labelsMat.deallocate();
        }
        System.out.println(
            "Successfully trained model using " + totalImgs + " images"
        );
        return faceRecognizer;
    }
}
