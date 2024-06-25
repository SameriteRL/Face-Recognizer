package raymond;

import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.opencv_core.Mat;

/**
 * Simple class to encapsulate facial feature and dimension data for facial
 * recognition.
 */
public class FrameData {

    // Bounding box dimensions
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
    public int label;
    public double predictScore;

    /**
     * Constructs a new FrameData using an OpenCV Mat. The associated label
     * and confidence score are set to -1 by default.
     * 
     * @param mat Mat to construct the frame with.
     */
    public FrameData(int row, Mat mat) {
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
        this.label        = -1;
        this.predictScore = -1.0;
    }
}
