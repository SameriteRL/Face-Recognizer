package raymond;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import raymond.service.FacePredictorService;
import raymond.service.FileService;
import raymond.service.ImageService;

@SpringBootTest
class FaceRecognizerAppTest {

    @Autowired
    public FacePredictorService facePredictor;

    @Autowired
    public ImageService imageService;

    @Autowired
    public FileService fileService;

    @Test
    public void test() throws Exception {
        
    }
}
