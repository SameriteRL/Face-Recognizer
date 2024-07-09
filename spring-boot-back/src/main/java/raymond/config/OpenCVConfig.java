package raymond.config;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;

import org.bytedeco.opencv.opencv_core.Size;
import org.bytedeco.opencv.opencv_objdetect.FaceDetectorYN;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Scope;

@Configuration
public class OpenCVConfig {

    @Value("${app.service.facerecognizerpath}")
    private String faceRecognizerModelPath;

    @Value("${app.service.facedetectorpath}")
    private String faceDetectorModelPath;

    private FaceRecognizerSF faceRecognizer;

    @PostConstruct
    public void init() {
        faceRecognizer =
            FaceRecognizerSF.create(faceRecognizerModelPath, "");
    }

    @PreDestroy
    public void cleanup() {
        if (faceRecognizer != null) {
            faceRecognizer.close();
        }
    }

    @Bean
    @Scope("singleton")
    public FaceRecognizerSF faceRecognizer() {
        return faceRecognizer;
    }

    @Bean
    @Scope("prototype")
    public FaceDetectorYN faceDetectorYN() {
        try (Size inputSize = new Size()) {
            return FaceDetectorYN.create(faceDetectorModelPath, "", inputSize);
        }
    }
}
