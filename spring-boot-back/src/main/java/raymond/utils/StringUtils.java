package raymond.utils;

public class StringUtils {
    
    /**
     * Returns the file extension of the given path, not including the
     * extension separator. If the path is a directory or otherwise has no file
     * extension, an empty string is returned. <p>
     * 
     * The extension is the portion at the end of the file's name after the
     * dot. For example, the extension of {@code "myfolder/doc.txt"} is
     * {@code "txt"}.
     * 
     * @param path Path to get the extension of.
     * @return The extension of the path, or an empty string if no extension
     *         exists.
     * @throws NullPointerException If the path is null.
     */
    public static String getExtension(String path) {
        if (path == null) {
            throw new NullPointerException("Path is null");
        }
        int extSepIdx = path.lastIndexOf('.');
        if (extSepIdx == -1) {
            return "";
        }
        return path.substring(extSepIdx + 1);
    }

    /**
     * Returns the file extension of the given path, including the extension
     * dot separator. If the path is a directory or otherwise has no file
     * extension, an empty string is returned. <p>
     * 
     * The extension is the portion at the end of the file's name after the
     * dot. For example, the extension of {@code "myfolder/doc.txt"} including
     * the dot is {@code ".txt"}.
     * 
     * @param path Path to get the extension of.
     * @return The extension of the path with the dot separator, or an empty
     *         string if no extension exists.
     * @throws NullPointerException If the path is null.
     */
    public static String getExtensionWithDot(String path) {
        if (path == null) {
            throw new NullPointerException("Path is null");
        }
        int extSepIdx = path.lastIndexOf('.');
        if (extSepIdx == -1) {
            return "";
        }
        return path.substring(extSepIdx);
    }

    /**
     * Returns the file stem of the given path. If the path is a directory or
     * otherwise has no file extension, the path's basename is returned.
     * <p>
     * 
     * The stem is the file's name not including its suffix. For example, the
     * stem of {@code "myfolder/doc.txt"} is {@code "doc"}.
     * <p>
     * 
     * The basename is the same thing as the file's name. For example, the
     * basename of {@code "myfolder/doc.txt"} is {@code "doc.txt"}.
     * 
     * @param path Path to get the stem of.
     * @return The stem of the path, which is also the basename of the file if
     *         it has no extension.
     * @throws NullPointerException If the path is null.
     */
    public static String getStem(String path) {
        if (path == null) {
            throw new NullPointerException("Path is null");
        }
        int lastSepIdx = path.lastIndexOf('/');
        String baseName = path.substring(lastSepIdx + 1);
        int extSepIdx = baseName.lastIndexOf('.');
        // For directory paths like "folder/insidefolder/"
        if (baseName.isEmpty()) {
            return path.substring(
                path.lastIndexOf('/', lastSepIdx - 1),
                lastSepIdx
            );
        }
        if (extSepIdx == -1) {
            return baseName;
        }
        return baseName.substring(0, extSepIdx);
    }
}
