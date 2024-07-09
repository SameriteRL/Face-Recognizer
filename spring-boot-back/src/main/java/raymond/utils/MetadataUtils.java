package raymond.utils;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.drew.imaging.FileTypeDetector;
import com.drew.imaging.ImageMetadataReader;
import com.drew.imaging.ImageProcessingException;
import com.drew.metadata.Directory;
import com.drew.metadata.Metadata;
import com.drew.metadata.MetadataException;
import com.drew.metadata.Tag;
import com.drew.metadata.exif.ExifIFD0Directory;
import com.drew.metadata.exif.ExifSubIFDDirectory;
import com.drew.metadata.jpeg.JpegDirectory;
import com.drew.metadata.png.PngDirectory;

public class MetadataUtils {

    /**
     * Gets EXIF orientation data from a media file.
     * 
     * @param mediaPath Path of the image file.
     * @return The EXIF orientation data tag value of the media, or {@code -1}
     *         if it doesn't exist or can't be determined.
     * @throws NullPointerException If the media path is null.
     */
    public static int getExifOrientation(String mediaPath) {
        Objects.requireNonNull(mediaPath, "Media path");
        try {  
            File mediaFile = new File(mediaPath);
            Metadata md = ImageMetadataReader.readMetadata(mediaFile);
            Directory exifDir =
                md.getFirstDirectoryOfType(ExifIFD0Directory.class);
            if (exifDir == null) {
                exifDir = md.getFirstDirectoryOfType(ExifSubIFDDirectory.class);
            }
            // Orientation tag code
            return exifDir.getInt(274);
        }
        catch (IOException | ImageProcessingException | MetadataException exc) {
            return -1;
        }
    }

    /**
     * @param mediaBytes Byte array of the media file.
     * @return The detected file type of the media file (e.g. JPEG, PNG), or
     *         {@code null} if an error occurs during reading or if the type
     *         cannot be determined.
     * @throws NullPointerException If the media byte array is null.
     */
    public static String getFileType(byte[] mediaBytes) {
        Objects.requireNonNull(mediaBytes, "Media byte array");
        try (InputStream inStream = new BufferedInputStream(
                new ByteArrayInputStream(mediaBytes)
        )) {
            return FileTypeDetector.detectFileType(inStream).getName();
        }
        catch (IOException exc) {
            return null;
        }
    }

    /**
     * @param mediaPath Path of the media file.
     * @return The detected file type of the media file (e.g. JPEG, PNG), or
     *         {@code null} if an error occurs during reading or if the type
     *         cannot be determined.
     * @throws NullPointerException If the media path is null.
     */
    public static String getFileType(String mediaPath) {
        Objects.requireNonNull(mediaPath, "Media path");
        try (InputStream inStream = new BufferedInputStream(
                new FileInputStream(new File(mediaPath))
        )) {
            return FileTypeDetector.detectFileType(inStream).getName();
        }
        catch (IOException exc) {
            return null;
        }
    }

    /**
     * @param mediaBytes Byte array of the media file.
     * @return A two-element array <code>{width, height}</code>, or
     *         {@code null} if the width and height could not be determined
     *         from the metadata.
     * @throws NullPointerException If the byte array is null.
     */
    public static int[] getWidthHeight(byte[] mediaBytes) {
        Objects.requireNonNull(mediaBytes);
        try (InputStream inStream = new ByteArrayInputStream(mediaBytes)) {
            Metadata md = ImageMetadataReader.readMetadata(inStream);
            return getWidthHeight(md);
        }
        catch (IOException | ImageProcessingException exc) {
            return null;
        }
    }

    /**
     * @param mediaPath Path of the media file.
     * @return A two-element array <code>{width, height}</code>, or
     *         {@code null} if the width and height could not be determined
     *         from the metadata.
     * @throws NullPointerException If the media path is null.
     */
    public static int[] getWidthHeight(String mediaPath) {
        Objects.requireNonNull(mediaPath, "Media path");
        try {
            File mediaFile = new File(mediaPath);
            Metadata md = ImageMetadataReader.readMetadata(mediaFile);
            return getWidthHeight(md);
        }
        catch (IOException | ImageProcessingException exc) {
            return null;
        }
    }

    /**
     * @param md Metadata of the media file.
     * @return A two-element array <code>{width, height}</code>, or
     *         {@code null} if the width and height could not be determined
     *         from the metadata.
     * @throws NullPointerException If the metadata argument is null.
     */
    public static int[] getWidthHeight(Metadata md) {
        Objects.requireNonNull(md, "Metadata object");
        int[] widthHeight = new int[2];
        for (JpegDirectory jpegDir: md.getDirectoriesOfType(JpegDirectory.class)) {
            try {
                widthHeight[0] = jpegDir.getImageWidth();
                widthHeight[1] = jpegDir.getImageHeight();
                return widthHeight;
            }
            catch (MetadataException exc) {}
        }
        for (PngDirectory pngDir: md.getDirectoriesOfType(PngDirectory.class)) {
            try {
                widthHeight[0] = pngDir.getInt(PngDirectory.TAG_IMAGE_WIDTH);
                widthHeight[1] = pngDir.getInt(PngDirectory.TAG_IMAGE_HEIGHT);
                return widthHeight;
            }
            catch (MetadataException exc) {}
        }
        for (ExifIFD0Directory exifDir: md.getDirectoriesOfType(ExifIFD0Directory.class)) {
            try {
                widthHeight[0] = exifDir.getInt(ExifIFD0Directory.TAG_IMAGE_WIDTH);
                widthHeight[1] = exifDir.getInt(ExifIFD0Directory.TAG_IMAGE_HEIGHT);
                return widthHeight;
            }
            catch (MetadataException exc) {}
        }
        for (ExifSubIFDDirectory exifDir: md.getDirectoriesOfType(ExifSubIFDDirectory.class)) {
            try {
                widthHeight[0] = exifDir.getInt(ExifSubIFDDirectory.TAG_IMAGE_WIDTH);
                widthHeight[1] = exifDir.getInt(ExifSubIFDDirectory.TAG_IMAGE_HEIGHT);
                return widthHeight;
            }
            catch (MetadataException exc) {}
        }
        return null;
    }
    
    /**
     * @param mediaBytes Byte array of the media file.
     * @return An immutable map with directory names as keys and their
     *         corresponding collections of tags as values. If metadata could
     *         not be read from the media, an empty immutable map is returned
     *         instead.
     * @throws NullPointerException If the media byte array is null.
     */
    public static Map<String, Collection<Tag>> getMetadata(byte[] mediaBytes) {
        Objects.requireNonNull(mediaBytes, "Media byte array");
        try {
            InputStream inStream = new ByteArrayInputStream(mediaBytes);
            Metadata metadata = ImageMetadataReader.readMetadata(inStream);
            return getMetadata(metadata);
        }
        catch (IOException | ImageProcessingException exc) {
            return Collections.emptyMap();
        }
    }

    /**
     * @param mediaPath Path of the media file.
     * @return An immutable map with directory names as keys and their
     *         corresponding collections of tags as values. If metadata could
     *         not be read from the media, an empty immutable map is returned
     *         instead.
     * @throws NullPointerException if the media path is null.
     */
    public static Map<String, Collection<Tag>> getMetadata(String mediaPath) {
        Objects.requireNonNull(mediaPath, "Media path");
        try {
            File mediaFile = new File(mediaPath);
            Metadata metadata = ImageMetadataReader.readMetadata(mediaFile);
            return getMetadata(metadata);
        }
        catch (IOException | ImageProcessingException exc) {
            return Collections.emptyMap();
        }
    }

    /**
     * @param md The metadata of the media file.
     * @return An immutable map with directory names as keys and their
     *         corresponding collections of tags as values.
     * @throws NullPointerException If the metadata argument is null.
     */
    public static Map<String, Collection<Tag>> getMetadata(Metadata md) {
        Objects.requireNonNull(md, "Metadata object");
        Map<String, Collection<Tag>> parsedMetadata = new HashMap<>();
        for (Directory dir: md.getDirectories()) {
            parsedMetadata.put(dir.getName(), dir.getTags());
        }
        return Collections.unmodifiableMap(parsedMetadata);
    }
}
