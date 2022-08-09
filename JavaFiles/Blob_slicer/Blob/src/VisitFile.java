import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;

import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;

import static java.nio.file.FileVisitResult.CONTINUE;

public class VisitFile extends SimpleFileVisitor<Path> {

    // Print information about
    // each type of file.
    @Override
    public FileVisitResult visitFile (Path file,
                                      BasicFileAttributes attr) {

        if( attr.isRegularFile() & file.getFileName().toString().endsWith(".java")){
            Blob.FileContext="";
            //          Blob.fp=null;
            Blob.FileContext= readfile.readfile(file);
//            Blob.fp=file;
            //System.out.println("class VisitFile filename= "+file.toString());
            CharStream input = CharStreams.fromString(Blob.FileContext);
            JavaLexer javalexer = new JavaLexer(input);
            CommonTokenStream token = new CommonTokenStream(javalexer);


            JavaParser parser = new JavaParser(token);
            JavaParserBaseVisitor<String> visitor = new JavaParserBaseVisitor<String>();
            visitor.visit(parser.compilationUnit());
            Blob.java_file_counter++;
        }

        return CONTINUE;
    }

    // Print each directory visited.
    @Override
    public FileVisitResult postVisitDirectory(Path dir,
                                              IOException exc) {
        //  System.out.println("number of text files : "+slicer.txt_counter);
        //  System.out.println("number of java files : "+slicer.java_counter);
        // System.out.format("Directory: %s%n", dir);
        return CONTINUE;
    }

    // If there is some error accessing
    // the file, let the user know.
    // If you don't override this method
    // and an error occurs, an IOException
    // is thrown.
    @Override
    public FileVisitResult visitFileFailed(Path file,
                                           IOException exc) {
        System.err.println(exc);
        return CONTINUE;
    }
}
