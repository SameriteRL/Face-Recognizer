package raymond;

/**
 * Simple immutable class to encapsulate frame coordinate and size data.
 */
public class FrameData {

    public final int xCoord;
    public final int yCoord;
    public final int width;
    public final int height;

    /**
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
    }
}
