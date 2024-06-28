package raymond.controller;

import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import raymond.service.FaceRecognizerFacade;

@RestController
public class FileSubmitController {

    @Autowired
    private FaceRecognizerFacade faceRecognizerFacadeService;

    @PostMapping("/submit")
    public ResponseEntity<byte[]> handleImageSubmit(
        @RequestParam("faceImg") MultipartFile faceImg,
        @RequestParam("testImg") MultipartFile testImg
    ) throws IOException {
        try {
            return ResponseEntity.ok(
                faceRecognizerFacadeService.recognizeFaces(faceImg, testImg)
            );
        }
        catch (Exception exc) {
            exc.printStackTrace();
            return ResponseEntity.internalServerError().body(null);
        }
    }
}
