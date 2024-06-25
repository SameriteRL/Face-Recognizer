package raymond;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;

import com.drew.imaging.ImageMetadataReader;
import com.drew.imaging.ImageProcessingException;
import com.drew.metadata.Metadata;
import com.drew.metadata.MetadataException;
import com.drew.metadata.exif.ExifIFD0Directory;

/**
 * This class is a collection of image processing/manipulation methods not
 * closely related to the main facial detection/recognition logic.
 */
public class ImageUtils {

    /**
     * Creates and returns a new square version of the given Mat. <br></br>
     * 
     * The original Mat is not modified or deallocated as a result of the
     * resize; it is the responsibility of the caller to deallocate it
     * afterwards if they wish to.
     * 
     * @param src Mat to be resized.
     * @param length Desired length of the resized Mat.
     * @return A Mat that's been resized from the given Mat.
     * @throws NullPointerException If the Mat is null.
     */
    public static Mat squareMat(Mat src, int length) {
        if (src == null) {
            throw new NullPointerException("Source Mat is null");
        }
        Mat resizedMat = new Mat();
        Size size = new Size(length, length);
        resize(src, resizedMat, size);
        size.deallocate();
        return resizedMat;
    }

    /**
     * Creates a BufferedImage from an image path and corrects its orientation,
     * if necessary. <br></br>
     * 
     * Because ImageIO.read() does not read image metadata, photos that were
     * rotated on smartphones might be loaded in with their original
     * orientation. This function checks the image's orientation metadata and
     * corrects the image if necessary, resulting in a BufferedImage that
     * matches the orientation you see in the file system. <br></br>
     * 
     * If metadata cannot be determined from the image, an unmodified
     * BufferedImage is returned. <br></br>
     * 
     * @param imgPath Path to the image file.
     * @return BufferedImage with the correct orientation.
     * @throws NullPointerException If the image path is null.
     * @throws IOException If the stream is invalid.
     */
    public static BufferedImage createBufferedImage(
        String imgPath
    ) throws IOException {
        if (imgPath == null) {
            throw new NullPointerException("Image path is null");
        }
        File imgFile = new File(imgPath);
        BufferedImage img = ImageIO.read(imgFile);
        if (img == null) {
            throw new IOException("Stream could not be read as an image");
        }
        Metadata metadata = null;
        // Returns unmodified image if EXIF orientation cannot be determined
        try {
            metadata = ImageMetadataReader.readMetadata(imgFile);
        }
        catch (ImageProcessingException e) {
            return img;
        }
        ExifIFD0Directory exifIFD0 =
            metadata.getFirstDirectoryOfType(ExifIFD0Directory.class);
        if (exifIFD0 == null) {
            return img;
        }
        int orientation = -1;
        try {
            orientation = exifIFD0.getInt(ExifIFD0Directory.TAG_ORIENTATION);
        }
        catch (MetadataException e) {
            return img;
        }
        int rotateDegrees = -1;
        switch (orientation) {
            // Image is oriented normally
            case 1:
                return img;
            // Right side, top (rotate 90 degrees CW)
            case 6:
                rotateDegrees = 90;
                break;
            // Bottom, right side (rotate 180 degrees)
            case 3:
                rotateDegrees = 180;
                break;
            // Left side, bottom (rotate 270 degrees CW)
            case 8:
                rotateDegrees = 270;
                break;
        }
        // Rotates the image if orientation is incorrect
        int width = img.getWidth();
        int height = img.getHeight();
        BufferedImage rotatedImage = new BufferedImage(
            height,
            width,
            img.getType()
        );
        Graphics2D g2d = rotatedImage.createGraphics();
        AffineTransform transform = new AffineTransform();
        transform.translate(height / 2, width / 2);
        transform.rotate(Math.toRadians(rotateDegrees));
        transform.translate(-width / 2, -height / 2);
        g2d.setTransform(transform);
        g2d.drawImage(img, 0, 0, null);
        g2d.dispose();
        return rotatedImage;
    }

    /**
     * Draws a red rectangular frame onto the given image for each FrameData in
     * the given list. The image is modified in-place. <br></br>
     * 
     * The X and Y coordinates marks the top left corner of each frame. The
     * width corresponds to the frame's size along the positive X-axis, and the
     * length corresponds to the size along the negative Y-axis.
     * 
     * @param img Image to draw onto.
     * @param frameDataList List of FrameData objects describing the position
     *                      and dimensions of each frame to be drawn.
     * @throws NullPointerException If any arguments are null.
     */
    public static void drawFrames(
        BufferedImage img,
        List<FrameData> frameDataList,
        Map<Integer, String> labelLegend
    ) {
        drawFrames(img, frameDataList, labelLegend, false);
    }

    /**
     * Underlying method with an extra parameter to enable or disable debugging
     * mode. <br></br>
     * 
     * If <code>debug == true</code>, the confidence score associated with each
     * frame is also drawn. <br></br>
     * 
     * See {@link #drawFrames(BufferedImage, List)}.
     * 
     * @param img Image to draw onto.
     * @param frameDataList List of FrameData objects describing the position
     *                      and dimensions of each frame to be drawn.
     * @param labelLegend Map that maps each unique label to a corresponding
     *                    string name.
     * @param debug Enable or disable debug mode.
     * @throws NullPointerException If any arguments are null.
     */
    public static void drawFrames(
        BufferedImage img,
        List<FrameData> frameDataList,
        Map<Integer, String> labelLegend,
        boolean debug
    ) {
        if (img == null) {
            throw new NullPointerException("Image is null");
        }
        if (frameDataList == null) {
            throw new NullPointerException("Frame data list is null");
        }
        if (labelLegend == null) {
            throw new NullPointerException("Label legend is null");
        }
        Graphics2D g2d = img.createGraphics();
        g2d.setColor(Color.RED);
        // Stroke and font size proportional to length of img's smallest side
        int length = (img.getWidth() < img.getHeight())
            ? img.getWidth() : img.getHeight();
        int rectStrokeSize = length / 300;
        int fontSize = rectStrokeSize * 10;
        g2d.setStroke(
            new BasicStroke(
                rectStrokeSize,
                BasicStroke.CAP_ROUND,
                BasicStroke.JOIN_ROUND,
                3f,
                null,
                0f
            )
        );
        g2d.setFont(new Font("Arial", Font.PLAIN, fontSize));
        for (FrameData frameData: frameDataList) {
            // X and Y coords indicate top left of rectangle
            g2d.drawRect(
                frameData.xCoord,
                frameData.yCoord,
                frameData.width,
                frameData.height
            );
            // Labels are drawn directly above rectangle
            if (debug) {
                g2d.drawString(
                    String.format(
                        "%s : %.3f",
                        labelLegend.get(frameData.label),
                        frameData.predictScore
                    ),
                    frameData.xCoord,
                    frameData.yCoord - fontSize / 2
                );
            }
            else {
                g2d.drawString(
                    labelLegend.get(frameData.label),
                    frameData.xCoord,
                    frameData.yCoord - fontSize / 2
                );
            }
        }
        g2d.dispose();
    }
}
