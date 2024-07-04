package raymond.classes;

import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;

/**
 * Simple class that encapsulates coordinates, face feature, face detection,
 * and face recognition data related to a face's bounding box.
 */
public class FaceBox {

    // Bounding box data
    public final int xCoord;
    public final int yCoord;
    public final int width;
    public final int height;

    // Facial feature coordinates
    public final int xRightEye;
    public final int yRightEye;
    public final int xLeftEye;
    public final int yLeftEye;
    public final int xNoseTip;
    public final int yNoseTip;
    public final int xRightMouth;
    public final int yRightMouth;
    public final int xLeftMouth;
    public final int yLeftMouth;

    // Facial detection data
    public final float detectScore;
    
    // Facial recognition data
    public String label;
    public double predictScore;

    /**
     * Constructs a new FrameData using one row of a face detection result Mat,
     * typically determined using the YuNet face detection model. In other
     * words, the detection result of only one face. <p>
     * 
     * The associated label is set to an empty string and the associated
     * prediction score is set to -1 by default.
     * 
     * @param mat Face detection result row to construct the FaceBox with.
     */
    public FaceBox(Mat mat) {
        FloatRawIndexer indexer = mat.createIndexer();
        this.xCoord       = (int) indexer.get(0, 0);
        this.yCoord       = (int) indexer.get(0, 1);
        this.width        = (int) indexer.get(0, 2);
        this.height       = (int) indexer.get(0, 3);
        this.xRightEye    = (int) indexer.get(0, 4);
        this.yRightEye    = (int) indexer.get(0, 5);
        this.xLeftEye     = (int) indexer.get(0, 6);
        this.yLeftEye     = (int) indexer.get(0, 7);
        this.xNoseTip     = (int) indexer.get(0, 8);
        this.yNoseTip     = (int) indexer.get(0, 9);
        this.xRightMouth  = (int) indexer.get(0, 10);
        this.yRightMouth  = (int) indexer.get(0, 11);
        this.xLeftMouth   = (int) indexer.get(0, 12);
        this.yLeftMouth   = (int) indexer.get(0, 13);
        this.detectScore  =       indexer.get(0, 14);
        this.label        = "";
        this.predictScore = -1.0;
    }

    /**
     * Constructs a new FrameData using a face detection result Mat, typically
     * determined using the YuNet face detection model. <p>
     * 
     * The associated label is set to an empty string and the associated
     * prediction score is set to -1 by default.
     * 
     * @param mat Face detection result Mat to construct the FaceBox with.
     */
    public FaceBox(int row, Mat mat) {
        FloatRawIndexer indexer = mat.createIndexer();
        this.xCoord       = (int) indexer.get(row, 0);
        this.yCoord       = (int) indexer.get(row, 1);
        this.width        = (int) indexer.get(row, 2);
        this.height       = (int) indexer.get(row, 3);
        this.xRightEye    = (int) indexer.get(row, 4);
        this.yRightEye    = (int) indexer.get(row, 5);
        this.xLeftEye     = (int) indexer.get(row, 6);
        this.yLeftEye     = (int) indexer.get(row, 7);
        this.xNoseTip     = (int) indexer.get(row, 8);
        this.yNoseTip     = (int) indexer.get(row, 9);
        this.xRightMouth  = (int) indexer.get(row, 10);
        this.yRightMouth  = (int) indexer.get(row, 11);
        this.xLeftMouth   = (int) indexer.get(row, 12);
        this.yLeftMouth   = (int) indexer.get(row, 13);
        this.detectScore  =       indexer.get(row, 14);
        this.label        = "";
        this.predictScore = -1.0;
    }

    /**
     * Constructs a new FrameData using a face detection result Mat, typically
     * determined using the YuNet face detection model. The given scale factors
     * are applied to all X and Y coordinates in the Mat. <p>
     * 
     * The associated label is set to an empty string and the associated
     * prediction score is set to -1 by default.
     * 
     * @param mat Face detection result Mat to construct the FaceBox with.
     * @param scaleX Factor to scale X-coordinates by.
     * @param scaleY Factor to scale Y-coorindates by.
     */
    public FaceBox(int row, Mat mat, double scaleX, double scaleY) {
        FloatRawIndexer indexer = mat.createIndexer();
        this.xCoord       = (int) (indexer.get(row, 0)  * scaleX);
        this.yCoord       = (int) (indexer.get(row, 1)  * scaleY);
        this.width        = (int) (indexer.get(row, 2)  * scaleX);
        this.height       = (int) (indexer.get(row, 3)  * scaleY);
        this.xRightEye    = (int) (indexer.get(row, 4)  * scaleX);
        this.yRightEye    = (int) (indexer.get(row, 5)  * scaleY);
        this.xLeftEye     = (int) (indexer.get(row, 6)  * scaleX);
        this.yLeftEye     = (int) (indexer.get(row, 7)  * scaleY);
        this.xNoseTip     = (int) (indexer.get(row, 8)  * scaleX);
        this.yNoseTip     = (int) (indexer.get(row, 9)  * scaleY);
        this.xRightMouth  = (int) (indexer.get(row, 10) * scaleX);
        this.yRightMouth  = (int) (indexer.get(row, 11) * scaleY);
        this.xLeftMouth   = (int) (indexer.get(row, 12) * scaleX);
        this.yLeftMouth   = (int) (indexer.get(row, 13) * scaleY);
        this.detectScore  =        indexer.get(row, 14);
        this.label        = "";
        this.predictScore = -1.0;
    }
}
