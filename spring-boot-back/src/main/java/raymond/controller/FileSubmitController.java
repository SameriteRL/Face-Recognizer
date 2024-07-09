package raymond.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Base64Utils;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.IntNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.fasterxml.jackson.databind.node.TextNode;

import raymond.service.FaceRecognizerFacade;
import raymond.utils.MetadataUtils;

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
    ) {
        try {
            ObjectNode resp = objectMapper.createObjectNode();
            byte[] imgBytes =
                faceRecognizerFacadeService.recognizeFaces(faceImg, testImg);
            String b64encoding = Base64Utils.encodeToString(imgBytes);
            resp.set("result", new TextNode(b64encoding));
            resp.set("type", new TextNode(MetadataUtils.getFileType(imgBytes)));
            int[] widthHeight = MetadataUtils.getWidthHeight(imgBytes);
            resp.set("width", new IntNode(widthHeight[0]));
            resp.set("height", new IntNode(widthHeight[1]));
            return resp;
        }
        catch (Exception exc) {
            exc.printStackTrace();
            return null;
        }
    }
}
