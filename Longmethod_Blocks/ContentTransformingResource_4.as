Object#as#(#Class#clazz#)#{##if#(#Appendable#.#class#.#isAssignableFrom#(#clazz#)#)#{##if#(#isAppendSupported#(#)#)#{#final#Appendable#a#=#(#Appendable#)#getResource#(#)#.#as#(#Appendable#.#class#)#;##if#(#a#!=#null#)#{##return#new#Appendable#(#)#{#public#OutputStream#getAppendOutputStream#(#)#throws#IOException#{#OutputStream#out#=#a#.#getAppendOutputStream#(#)#;##if#(#out#!=#null#)#{#out#=#wrapStream#(#out#)#;#}##return#out#;#}#}#;#}#}##return#null#;#}##return#FileProvider#.#class#.#isAssignableFrom#(#clazz#)#?#null#:#getResource#(#)#.#as#(#clazz#)#;#}