public class blockValidation {

    public boolean valid(int index){
        int counter;
        // System.out.println("0 "+index);
        if(slicer.methodContext[index-1].equals("}") & !slicer.methodContext[index+1].equals("}") & !slicer.methodContext[index+1].equals("else") & !slicer.methodContext[index+1].equals(";") & !slicer.methodContext[index+1].equals(")"))
            return true;

        // System.out.println("1 "+index);
        if(index>2 & slicer.methodContext[index+1].equals("return"))
            //if(!slicer.methodContext[index-1].equals("{"))
               return true;

        if(index>2 & slicer.methodContext[index+1].equals("if"))
            return true;

        if( index > 2 &  slicer.methodContext[index-1].equals(";")  & !slicer.methodContext[index+1].equals("}") ){
            if(slicer.methodContext[index+1].equals("if") | slicer.methodContext[index+1].equals("while")| slicer.methodContext[index+1].equals("for") | slicer.methodContext[index+1].equals("switch") | slicer.methodContext[index+1].equals("try") | slicer.methodContext[index+1].equals("do") | slicer.methodContext[index+1].equals("synchronized"))
                return true;
            //   System.out.println("2 "+index);
            counter= index-3;
            while ( counter>1 & !slicer.methodContext[counter].equals(";") & !slicer.methodContext[counter].equals("{")  & !slicer.methodContext[counter].equals("}") & !slicer.methodContext[counter].equals("while") & !slicer.methodContext[counter].equals("for")& !slicer.methodContext[counter].equals("do") & !slicer.methodContext[counter].equals("default")  & !slicer.methodContext[counter].equals("if") & !slicer.methodContext[counter].equals("else") & !slicer.methodContext[counter].equals("return") ) {
                //     System.out.print("      c[counter]=" + Slicer.c[counter] + " counter= " + counter);
                counter -= 2;
            }
            if(slicer.methodContext[counter].equals("return") )
                return true;

            if( !slicer.methodContext[counter].equals(";") & !slicer.methodContext[counter].equals("{") & !slicer.methodContext[counter].equals("}") ){
//                System.out.println("3 "+index);
                return true;
            }
        }
        //      System.out.println("5 "+index);
        if(slicer.methodContext[index+1].length() >= 2)
            if( slicer.methodContext[index+1].substring(0,2).equals("//") | slicer.methodContext[index+1].substring(0,2).equals("/*") )
                if( !slicer.methodContext[index-1].equals("") )
                   // if(slicer.methodContext[index-1].length() >= 2){
                        //if( !slicer.methodContext[index-1].substring(0,2).equals("//") & !slicer.methodContext[index-1].substring(0,2).equals("/*") )
                            return true;
                    //}else{
                       // return true;
                    //}
        return false;
    }// valid method

}
