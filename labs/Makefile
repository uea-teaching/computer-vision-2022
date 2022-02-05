PDF := $(patsubst %.md,%.lab.pdf,$(wildcard *.md))

all : $(PDF)

%.lab.pdf : %.md
	pandoc $^ \
	-M "figPrefix:" \
	--filter pandoc-crossref \
	--citeproc \
	--bibliography refs.bib \
	-V geometry:left=3cm,right=3cm,top=2cm,bottom=3cm \
	-t pdf \
	-o $@
