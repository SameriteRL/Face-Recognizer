package raymond;

import java.util.Collection;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;

import com.drew.metadata.Tag;

import raymond.utils.MetadataUtils;

@SpringBootTest
class FaceRecognizerAppTest {

    private static final String testImgPath =
        "/Users/raymond/Pictures/frieren campfire.png";

    @Test
    public void testGetMetadata() {
        Map<String, Collection<Tag>> metadata =
            MetadataUtils.getMetadata(testImgPath);
        for (String dirName: metadata.keySet()) {
            System.out.println(dirName);
            for (Tag tag: metadata.get(dirName)) {
                System.out.println("\t" + tag);
            }
        }
    }

    @Test
    public void testGetMediaFileType() {
        System.out.println(MetadataUtils.getFileType(testImgPath));
    }
}
