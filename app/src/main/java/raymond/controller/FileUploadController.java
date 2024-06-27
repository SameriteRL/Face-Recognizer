package raymond.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.FileOutputStream;

@RestController
public class FileUploadController {

    @PostMapping("/upload")
    public ResponseEntity<String> handleFileUpload(
        @RequestParam("faceImg") MultipartFile faceImg,
        @RequestParam("testImg") MultipartFile testImg
    ) {
        try (FileOutputStream outStr = new FileOutputStream("out.png")) {
            outStr.write(faceImg.getBytes());
        }
        catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                                 .body("Failed to upload file");
        }
        return ResponseEntity.ok("File received");
    }
}
