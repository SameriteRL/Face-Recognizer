package raymond;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;
import java.util.Map;

import javax.imageio.ImageIO;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_objdetect.FaceRecognizerSF;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import raymond.classes.ROIData;
import raymond.service.FacePredictor;
import raymond.utils.ImageUtils;
import raymond.utils.StringUtils;

@SpringBootTest
class FaceRecognizerAppTest {

	@Test
	public void test() throws Exception {
		final String testImgPath = "test_images/skating.jpg";
		final String knownFacesDirPath = "known_faces";
		final String detectorModelPath = "models/yunet_detection_2023mar.onnx";
		final String recognizerModelPath = "models/sface_recognition_2021dec.onnx";
		final String testImgFormat = StringUtils.getExtension(testImgPath);
		Map<String, List<Mat>> knownFaces = FacePredictor.parseKnownFaces(
			knownFacesDirPath,
			detectorModelPath,
			recognizerModelPath
		);
		List<ROIData> roiList = null;
		try (FaceRecognizerSF faceRecognizer = 
				FaceRecognizerSF.create(recognizerModelPath, "")
		) {
			roiList = FacePredictor.predictFaces(
				testImgPath,
				knownFaces,
				detectorModelPath,
				faceRecognizer
			);
		}
		BufferedImage bufImg = ImageUtils.createBufferedImage(testImgPath);
		ImageUtils.drawFrames(bufImg, roiList, true);
		File outImgFile = new File("out." + testImgFormat);
		ImageIO.write(bufImg, testImgFormat, outImgFile);
	}
}
