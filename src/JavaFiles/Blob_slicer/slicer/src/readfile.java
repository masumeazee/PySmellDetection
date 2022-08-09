import java.io.File;
import java.io.FileReader;
import java.nio.file.Path;

public class readfile {

    public static String readfile(Path file) {
        String filecontext="";
        try {
            File fp = new File(file.toString());
            FileReader fr = new FileReader(fp);
            char[] buf = new char[4096];
            int counter = 0;
            int i;

            counter = fr.read(buf);
            while (counter > 0) {
                for (i = 0; i < counter; i++) {
                    filecontext += buf[i];
                }
                counter = fr.read(buf);
            }
        }catch (Exception e){}
        return filecontext;
    }
}
