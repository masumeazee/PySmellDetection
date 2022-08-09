import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class slicer {
    public static final int MX = 50000;
    public static Integer java_counter = 0;
    public static String fileContext = "";
    public static Path fileName;
    public static String[] method_list = new String[MX];
    public static Integer method_in_class_counter = 0;
    public static String[] methodContext = new String[MX];
    public static String methodName;
    public static int methodNumbers = 0;
    public static String methodTemporaryContext = "";
    public static int size = 0;

    public static void main(String[] args) throws Exception {
        VisitFile vf = new VisitFile();
        Path start = FileSystems.getDefault().getPath("E:", "long_methods", "test");
        Files.walkFileTree(start, vf);

        System.out.println("\nnumber of java files= " + java_counter + "  number of methods= " + methodNumbers+"\nThe End");
    }


}
