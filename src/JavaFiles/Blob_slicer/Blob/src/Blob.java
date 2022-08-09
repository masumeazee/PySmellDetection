import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;


public class Blob {

    public static final int MX = 50000;
    public static String[] methodContext = new String[MX];
    public static String FileContext;
    public static String methodTemporaryContext;
    public static Integer java_file_counter=0;
    public static Integer class_counter=0;
    public static String ClassName;
    public static String packageName;
    //   public static Path fp;

    public static void main(String[] args) throws Exception {
        VisitFile vf = new VisitFile();
        Path start = FileSystems.getDefault().getPath("E:", "long_methods", "test","Blob");
        Files.walkFileTree(start, vf);
        System.out.println("\nnumber of java files= " + java_file_counter + "  number of classes= " + class_counter+"\nThe End");
    }

}
