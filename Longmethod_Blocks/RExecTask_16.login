void#login#(#AntRExecClient#rexec#)#{##if#(#addCarriageReturn#)#{#rexec#.#sendString#(#"\n"#,#true#)#;#}##rexec#.#waitForString#(#"ogin:"#)#;#rexec#.#sendString#(#userid#,#true#)#;#rexec#.#waitForString#(#"assword:"#)#;#rexec#.#sendString#(#password#,#false#)#;#}