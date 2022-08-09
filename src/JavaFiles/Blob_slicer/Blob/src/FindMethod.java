import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.misc.Interval;

import javax.imageio.IIOException;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class FindMethod {
    public static void findmethod(){

        CharStream input = CharStreams.fromString(Blob.FileContext);
        //Java9Lexer javalexer = new Java9Lexer(input);
        //Java8Lexer javalexer = new Java8Lexer(input);
        JavaLexer javalexer = new JavaLexer(input);
        CommonTokenStream token = new CommonTokenStream(javalexer);
        Interval interval = new Interval(0,0);
        Interval interval1 = new Interval(0,0);
        boolean isfound=false;
        String temp="";
        int i;


        while(token.getText(interval1).startsWith("//") | token.getText(interval1).startsWith("/*")) {
            interval1.a++;
            interval1.b=interval1.a;
        }
        temp=token.getText(interval1);
        while(!token.getText(interval1).matches("") & !isfound ) {
            if(temp.length() < Blob.methodTemporaryContext.length()) {
                while (temp.equals(Blob.methodTemporaryContext.substring(0, temp.length())) & !isfound & !token.getText(interval1).matches("")) {
                    if (temp.length() == Blob.methodTemporaryContext.length()) {
                        isfound = true;
                    } else {
                        interval1.b++;
                        interval1.a = interval1.b;
                        while (token.getText(interval1).startsWith("//") | token.getText(interval1).startsWith("/*")) {
                            interval1.a++;
                            interval1.b = interval1.a;
                        }
                        interval.b = interval1.a;
                        temp += token.getText(interval1);

                        if (temp.length() > Blob.methodTemporaryContext.length())
                            break;
                    }
                }//while
            }//if
            if( !isfound ) {
                interval.a++;
                interval1.a=interval1.b=interval.b = interval.a;
                while(token.getText(interval1).startsWith("//") | token.getText(interval1).startsWith("/*")) {
                    interval.a++;
                    interval1.b = interval1.a = interval.b = interval.a;
                }
                temp = token.getText(interval1);
            }
        }
        if(isfound) {
            for (i = 0; i < Blob.MX; i++)
                Blob.methodContext[i] = "";

            interval1.a = interval.a;
            interval1.b = interval.a;
            for (i = 0; i < (interval.b - interval.a); i++) {
                Blob.methodContext[2 * i] = token.getText(interval1);
                Blob.methodContext[2 * i + 1] = "#";
                interval1.a++;
                interval1.b = interval1.a;
            }
            Blob.methodContext[2 * i] = token.getText(interval1);
            Blob.methodContext[2 * i + 1] = "##";


            try {
                File fp = new File("/blocked_files/blob/"+Blob.packageName+"."+Blob.ClassName+".txt");
                FileWriter w = new FileWriter(fp,true);
                for (i = 0; !(Blob.methodContext[i].equals("")); i++) {
                    w.write(Blob.methodContext[i]);
                }
                w.close();
            }catch (IOException e){e.printStackTrace();}
        }//if



    }//method
}//class
