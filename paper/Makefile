.PHONY: pdf clean
pdf:
	latexmk -pdf main.tex
clean:
	latexmk -C 2>/dev/null || rm -f main.aux main.bbl main.blg main.log main.out main.pdf main.fls main.fdb_latexmk
