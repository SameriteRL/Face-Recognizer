package raymond.service;

import java.io.File;
import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

@Service
public class FileService {

    @Autowired
    public ResourceLoader resourceLoader;

    public File getResourceFile(String resourcePath) throws IOException {
        Resource resource =
            resourceLoader.getResource("classpath:" + resourcePath);
        if (!resource.exists()) {
            throw new IOException("Resource path does not exist");
        }
        return resource.getFile();
    }
}
