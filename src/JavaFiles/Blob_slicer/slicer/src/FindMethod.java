import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.misc.Interval;

public class FindMethod {
    public static void findmethod(){
        CharStream input = CharStreams.fromString(slicer.fileContext);
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
            if(temp.length() < slicer.methodTemporaryContext.length()) {
                while (temp.equals(slicer.methodTemporaryContext.substring(0, temp.length())) & !isfound & !token.getText(interval1).matches("")) {
                    if (temp.length() == slicer.methodTemporaryContext.length()) {
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

                        if (temp.length() > slicer.methodTemporaryContext.length())
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
            for (i = 0; i < slicer.MX; i++)
                slicer.methodContext[i] = "";

            interval1.a = interval.a;
            interval1.b = interval.a;
            for (i = 0; i < (interval.b - interval.a); i++) {
                slicer.methodContext[2 * i] = token.getText(interval1);
                slicer.methodContext[2 * i + 1] = "##";
                interval1.a++;
                interval1.b = interval1.a;
            }
            slicer.methodContext[2 * i] = token.getText(interval1);
            Segmenter.segmenter();
        }//if

    }//method
}//class

