import java.io.File;
import java.io.FileWriter;

public class Segmenter {
    public static void segmenter(){

        blockValidation Valid = new blockValidation();
        for ( int i = 1; !slicer.methodContext[i].equals("") ; i+=2 )
            if (!Valid.valid(i))
                slicer.methodContext[i] = "#";
        try {

            String fn;
            Integer overlap=slicer.method_in_class_counter;
            fn=slicer.fileName.getFileName().toString().substring(0,slicer.fileName.getFileName().toString().length()-5)+"_"+overlap.toString()+"."+slicer.methodName;
            for (int i = 0; i < slicer.methodNumbers; i++) {
                    if (slicer.method_list[i].equals(fn)) {
                          fn=null;
                          overlap++;
                          fn=slicer.fileName.getFileName().toString().substring(0,slicer.fileName.getFileName().toString().length()-5)+"_"+overlap.toString()+"."+slicer.methodName;
                          i=-1;
                    }
            }
            slicer.method_list[slicer.methodNumbers]=fn;
            slicer.methodNumbers++;

            File fp = new File("e:/long_methods/blocked_files/"+fn);
            FileWriter w = new FileWriter(fp);
            for (int i = 0; !(slicer.methodContext[i].equals("")); i++){
                w.write(slicer.methodContext[i]);
            }
            w.close();



            /*
            for(int i=0; i <= slicer.MX ;i++){
                if(slicer.method_list[i].equals(slicer.fileName.toString()+"_"+slicer.method_in_class_counter.toString()+"."+slicer.methodName)){
                    slicer.method_list[i]="E";
                    break;
                }

            }
*/

        } catch (Exception e) {
        }


    }
}
