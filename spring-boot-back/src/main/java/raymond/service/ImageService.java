package raymond.service;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Objects;

import javax.imageio.ImageIO;

import org.springframework.stereotype.Service;

import raymond.classes.FaceBox;
import raymond.utils.MetadataUtils;
import raymond.utils.StringUtils;

@Service
public class ImageService {

    /**
     * Creates a BufferedImage and corrects its orientation, if necessary. <p>
     * 
     * {@code ImageIO.read()} does not read image metadata, so the
     * BufferedImage created from a camera image may be oriented incorrectly.
     * This method attempts to ensure correct orientation by reading the
     * image's EXIF metadata.
     * 
     * @param imgBytes Byte array of the image file.
     * @return A correctly oriented BufferedImage if EXIF metadata was
     *         successfully read. Otherwise, the unmodified BufferedImage
     *         created from {@code ImageIO.read()} is returned.
     * @throws NullPointerException If the image byte array is null.
     * @throws IOException If the image is invalid.
     */
    public BufferedImage createBufferedImage(
        byte[] imgBytes
    ) throws IOException {
        Objects.requireNonNull(imgBytes, "Image byte array");
        BufferedImage bufImg = null;
        try (InputStream bais = new ByteArrayInputStream(imgBytes)) {
            bufImg = ImageIO.read(bais);
        }
        if (bufImg == null) {
            throw new IOException("Invalid image stream");
        }
        int orientation = MetadataUtils.getExifOrientation(imgBytes);
        return correctOrientation(bufImg, orientation);
    }

    /**
     * Creates a BufferedImage and corrects its orientation, if necessary. <p>
     * 
     * {@code ImageIO.read()} does not read image metadata, so the
     * BufferedImage created from a camera image may be oriented incorrectly.
     * This method attempts to ensure correct orientation by reading the
     * image's EXIF metadata.
     * 
     * @param imgPath Path of the image file.
     * @return A correctly oriented BufferedImage if EXIF metadata was
     *         successfully read. Otherwise, the unmodified BufferedImage
     *         created from {@code ImageIO.read()} is returned.
     * @throws NullPointerException If the image path is null.
     * @throws IOException If the image is invalid.
     */
    public BufferedImage createBufferedImage(
        String imgPath
    ) throws IOException {
        Objects.requireNonNull(imgPath, "Image path");
        File imgFile = new File(imgPath);
        BufferedImage bufImg = ImageIO.read(imgFile);
        if (bufImg == null) {
            throw new IOException("Invalid image file");
        }
        int orientation = MetadataUtils.getExifOrientation(imgPath);
        return correctOrientation(bufImg, orientation);
    }

    /**
     * Helper method to correct the orientation of a BufferedImage by
     * evalutaing the given EXIF orientation value and rotating the image
     * accordingly.
     * 
     * @param src BufferedImage to be corrected.
     * @param exifOrientation Value of the image's EXIF orientation tag.
     * @return A new correctly oriented BufferedImage. If no correction is
     *         needed, the given BufferedImage is returned as-is.
     * @throws NullPointerException If the image path is null.
     * @throws IOException If the image is invalid.
     */
    public BufferedImage correctOrientation(
        BufferedImage src,
        int exifOrientation
    ) {
        int rotateDegrees = -1;
        switch (exifOrientation) {
            // Orientation data cannot be found or image is oriented normally
            case -1 | 1:
                return src;
            // Left side on bottom (rotate 90 degrees CW)
            case 6:
                rotateDegrees = 90;
                break;
            // Upside-down (rotate 180 degrees)
            case 3:
                rotateDegrees = 180;
                break;
            // Right side on bottom (rotate 270 degrees CW)
            case 8:
                rotateDegrees = 270;
                break;
        }
        // Rotates the image if orientation is incorrect
        int width = src.getWidth();
        int height = src.getHeight();
        BufferedImage rotatedImage = new BufferedImage(
            height,
            width,
            src.getType()
        );
        Graphics2D g2d = rotatedImage.createGraphics();
        AffineTransform transform = new AffineTransform();
        transform.translate(height / 2, width / 2);
        transform.rotate(Math.toRadians(rotateDegrees));
        transform.translate(-width / 2, -height / 2);
        g2d.setTransform(transform);
        g2d.drawImage(src, 0, 0, null);
        g2d.dispose();
        return rotatedImage;
    }

    /**
     * Visualizes face recognition results by drawing bounding boxes onto a
     * test image. Returns a byte array representation of the resulting image.
     * 
     * @param imgPath Path of the image to draw onto.
     * @param faceBoxList List of face boxes to draw onto the image.
     * @return A byte array representing the resulting image.
     * @throws NullPointerException If any arguments are null.
     * @throws IOException If a format cannot be determined from the image file
     *                     name, or for general I/O errors.
     */
    public byte[] visualizeBoxes(
        String imgPath,
        List<FaceBox> faceBoxList
    ) throws IOException {
        Objects.requireNonNull(imgPath, "Image path");
        Objects.requireNonNull(faceBoxList, "Face box list");
        String imgFormat = StringUtils.getExtension(imgPath);
        BufferedImage bufImg = createBufferedImage(imgPath);
        drawBoxes(bufImg, faceBoxList, false);
        ByteArrayOutputStream imgOutStream = new ByteArrayOutputStream();
        ImageIO.write(bufImg, imgFormat, imgOutStream);
        return imgOutStream.toByteArray();
    }

    /**
     * Helper method to draw red rectangular boxes onto an image for each ROI
     * in the given list. The image is modified in-place. <p>
     * 
     * The X and Y coordinates marks the top left corner of each frame. The
     * width corresponds to the frame's size along the positive X-axis, and the
     * length corresponds to the size along the negative Y-axis. <p>
     * 
     * If {@code debug == true}, the "confidence" score associated with each
     * frame is also drawn.
     * 
     * @param img Image to draw onto.
     * @param faceBoxList List of face boxes.
     * @param debug Enable or disable debug mode.
     * @throws NullPointerException If any arguments are null.
     */
    private void drawBoxes(
        BufferedImage img,
        List<FaceBox> faceBoxList,
        boolean debug
    ) {
        Objects.requireNonNull(img, "Input image");
        Objects.requireNonNull(faceBoxList, "ROI list");
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
        for (FaceBox roi: faceBoxList) {
            // X and Y coords indicate top left of rectangle
            g2d.drawRect(
                roi.xCoord,
                roi.yCoord,
                roi.width,
                roi.height
            );
            // Labels are drawn directly above rectangle
            if (debug) {
                g2d.drawString(
                    String.format("%s : %.3f", roi.label, roi.predictScore),
                    roi.xCoord,
                    roi.yCoord - fontSize / 2
                );
                continue;
            }
            g2d.drawString(roi.label, roi.xCoord, roi.yCoord - fontSize / 2);
        }
        g2d.dispose();
    }
}
