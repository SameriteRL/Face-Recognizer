package raymond;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import com.drew.imaging.ImageMetadataReader;
import com.drew.imaging.ImageProcessingException;
import com.drew.metadata.Metadata;
import com.drew.metadata.MetadataException;
import com.drew.metadata.exif.ExifIFD0Directory;

public class FaceDetector {

    /**
     * Overload that uses the default Haar Cascade Frontal-Face model to
     * detect faces.
     * 
     * @see #detectFaces(String, String, String)
     * 
     * @param imgPath Path of the image to extract ROIs from.
     * @return A list of FrameData objects, each one representing a ROI.
     * @throws IOException If the image path is invalid, or for general I/O
     *                     errors.
     */
    public static List<FrameData> detectFaces(String imgPath) throws IOException {
        return detectFaces(
            imgPath,
            "models/haarcascade_frontalface_default.xml"
        );
    }

    /**
     * Performs facial detection on an image using the given cascade classifier
     * model and returns a list of FrameData objects, where each one represents
     * a region of interest (ROI). Returns an empty list if no ROIs are
     * detected.
     * 
     * @param imgPath Path of the image to extract ROIs from.
     * @param cascadeModelPath Path of the cascade model to use.
     * @return A list of FrameData objects, each one representing a ROI.
     * @throws IOException If the image or model paths are invalid, or for
     *                     general I/O errors.
     */
    public static List<FrameData> detectFaces(
        String imgPath,
        String cascadeModelPath
    ) throws IOException {
        File imgFile = new File(imgPath);
        File cascadeModelFile = new File(cascadeModelPath);
        BufferedImage bufImg = ImageIO.read(imgFile);
        if (bufImg == null) {
            throw new IOException("Image path is invalid or cannot be read");
        }
        if (!cascadeModelFile.exists() || cascadeModelFile.isDirectory()) {
            throw new IOException("Cascade model path is invalid");
        }
        // Face detection
        Mat imgMat = imread(imgPath);
        RectVector detectedFaces = new RectVector();
        try (CascadeClassifier detector =
                new CascadeClassifier(cascadeModelPath)
        ) {
            detector.detectMultiScale(imgMat, detectedFaces);
        }
        catch (Exception e) {
            throw new RuntimeException("Cascade model is invalid");
        }
        List<FrameData> roiData = new ArrayList<>();
        for (int i = 0; i < detectedFaces.size(); ++i) {
            Rect face = detectedFaces.get(i);
            roiData.add(
                new FrameData(
                    face.x(),
                    face.y(),
                    face.width(),
                    face.height()
                )
            );
        }
        return roiData;
    }

    /**
     * Creates a BufferedImage from a file path and corrects its orientation,
     * if necessary. <br></br>
     * 
     * Because ImageIO.read() does not read image metadata, photos that were
     * rotated on smartphones might be loaded in with their original
     * orientation. This function checks the image's orientation metadata and
     * corrects the image if necessary, resulting in a BufferedImage that
     * matches the orientation you see in the file system. <br></br>
     * 
     * If metadata cannot be determined from the stream, an unmodified
     * BufferedImage is returned. <br></br>
     * 
     * This method does not close the provided InputStream after the read
     * operation has completed; it is the responsibility of the caller to close
     * the stream, if desired.
     * 
     * @param imgPath Path to the image file.
     * @return BufferedImage with the correct orientation.
     * @throws IOException If the stream is invalid.
     */
    public static BufferedImage createBufferedImage(
            String imgPath
    ) throws IOException {
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
}
