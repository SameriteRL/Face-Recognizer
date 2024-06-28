package raymond.service;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

@Service
public class FileService {

    @Autowired
    public ResourceLoader resourceLoader;

    /**
     * Retrieves an existing file from the project resources directory.
     * 
     * @param resourcePath Path of the file relative to the resource directory.
     * @return The requested file.
     * @throws FileNotFoundException If the file does not exist.
     * @throws IOException For general I/O errors.
     */
    public File getResourceFile(String resourcePath) throws IOException {
        Resource resource =
            resourceLoader.getResource("classpath:" + resourcePath);
        if (!resource.exists()) {
            throw new IOException("Resource path does not exist");
        }
        return resource.getFile();
    }

    /**
     * Determines whether the specified resource exists within the project
     * resources directory.
     * 
     * @param resourcePath Path of the file relative to the resource directory.
     * @return true if the resource exists, false otherwise.
     */
    public boolean resourceExists(String resourcePath) {
        Resource resource =
            resourceLoader.getResource("classpath:" + resourcePath);
        if (resource.exists()) {
            return true;
        }
        return false;
    }

    /**
     * Recursively deletes the specified directory, meaning that the directory
     * and all of its subdirectories and subfiles will be deleted. Use this
     * method at your own risk.
     * 
     * @param path Path of the directory to delete recursively.
     * @throws IOException For general I/O errors.
     */
    public void deleteDirRecursive(Path path) throws IOException {
        Files.walkFileTree(path, new SimpleFileVisitor<Path>() {
            @Override
            public FileVisitResult visitFile(
                Path file,
                BasicFileAttributes attrs
            ) throws IOException {
                Files.delete(file);
                return FileVisitResult.CONTINUE;
            }
            @Override
            public FileVisitResult postVisitDirectory(
                Path dir,
                IOException exception
            ) throws IOException {
                Files.delete(dir);
                return FileVisitResult.CONTINUE;
            }
        });
    }
}
