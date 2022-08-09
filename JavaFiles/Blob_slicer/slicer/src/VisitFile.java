import java.io.IOException;
import java.nio.file.FileVisitResult;
import java.nio.file.Path;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import static java.nio.file.FileVisitResult.*;

import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.misc.Interval;
import org.antlr.v4.runtime.tree.ParseTreeWalker;

public class VisitFile extends SimpleFileVisitor<Path> {

    // Print information about
    // each type of file.
    @Override
    public FileVisitResult visitFile (Path file,
                                      BasicFileAttributes attr) {

        if( attr.isRegularFile() & file.getFileName().toString().endsWith(".java")){
            slicer.fileContext="";
            slicer.fileName=null;
            //   slicer.fileName=file.getFileName().toString();
            slicer.fileName = file;
            slicer.fileContext= readfile.readfile(file);
            //System.out.println("class VisitFile filename= "+file.toString());
            CharStream input = CharStreams.fromString(slicer.fileContext);
            //Java9Lexer javalexer = new Java9Lexer(input);
            //Java8Lexer javalexer = new Java8Lexer(input);
            JavaLexer javalexer = new JavaLexer(input);
            CommonTokenStream token = new CommonTokenStream(javalexer);


            //Java9Parser parser = new Java9Parser(token);
            // Java8Parser parser = new Java8Parser(token);
            JavaParser parser = new JavaParser(token);
            //Java9BaseListener listener= new Java9BaseListener();
            //Java8ParserBaseListener listener= new Java8ParserBaseListener();

            // JavaParserBaseListener listener= new JavaParserBaseListener();
            //ParseTreeWalker walker= new ParseTreeWalker();
            // walker.walk(listener,parser.classDeclaration());
            // walker.walk(listener,parser.compilationUnit());

            JavaParserBaseVisitor visitor = new JavaParserBaseVisitor();
            visitor.visit(parser.compilationUnit());

            slicer.java_counter++;
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
