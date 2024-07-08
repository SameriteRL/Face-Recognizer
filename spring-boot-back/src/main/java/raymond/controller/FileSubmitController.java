package raymond.controller;

import java.io.IOException;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Base64Utils;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import raymond.service.FaceRecognizerFacade;

@RestController
public class FileSubmitController {

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private FaceRecognizerFacade faceRecognizerFacadeService;

    @PostMapping("/submit")
    public ObjectNode handleImageSubmit(
        @RequestParam("faceImg") MultipartFile faceImg,
        @RequestParam("testImg") MultipartFile testImg
    ) throws IOException {
        ObjectNode map = objectMapper.createObjectNode();
        try {
            String b64encoding = Base64Utils.encodeToString(
                faceRecognizerFacadeService.recognizeFaces(faceImg, testImg)
            );
            map.put("result", b64encoding);
            return map;
        }
        catch (Exception exc) {
            exc.printStackTrace();
            return null;
        }
    }
}
