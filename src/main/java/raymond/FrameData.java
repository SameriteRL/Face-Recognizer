package raymond;

/**
 * Simple class to encapsulate frame coordinate, size, label, and confidence
 * data for facial recognition.
 */
public class FrameData {

    public final int xCoord;
    public final int yCoord;
    public final int width;
    public final int height;
    
    public int label;
    public double confidence;

    /**
     * Constructs a new FrameData with only frame coordinates and size. The
     * associated label and confidence score are set to -1 by default.
     * 
     * @param xCoord X-coordinate of the frame.
     * @param yCoord Y-coordinate of the frame.
     * @param width Length of frame along positive X-axis.
     * @param height Length of frame along negative Y-axis.
     */
    public FrameData(int xCoord, int yCoord, int width, int height) {
        this.xCoord = xCoord;
        this.yCoord = yCoord;
        this.width = width;
        this.height = height;
        this.label = -1;
        this.confidence = -1.0;
    }

    /**
     * Constructs a new FrameData with frame coordinates and size as well as
     * their associated label and confidence score.
     * 
     * @param xCoord X-coordinate of the frame.
     * @param yCoord Y-coordinate of the frame.
     * @param width Length of frame along positive X-axis.
     * @param height Length of frame along negative Y-axis.
     * @param label Label of frame.
     * @param confidence Confidence score of frame relative to label.
     */
    public FrameData(
        int xCoord,
        int yCoord,
        int width,
        int height,
        int label,
        double confidence
    ) {
        this.xCoord = xCoord;
        this.yCoord = yCoord;
        this.width = width;
        this.height = height;
        this.label = label;
        this.confidence = confidence;
    }
}
