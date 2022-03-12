HTM := $(patsubst %.md,%.html,$(wildcard *.md))
BMR := $(patsubst %.md,%.bmr.pdf,$(wildcard *.md))

all : $(HTM) $(BMR)

%.html : %.md
	pandoc $^ \
	--defaults revealjs \
	--css assets/style.css \
	-V highlightjs-theme=monokai \
	-V theme=black \
	-V transition=none \
	-V width=1200 \
	-V height=900 \
	-V margin=0.1 \
	-o $@

%.bmr.pdf : %.md
	pandoc 	$^ \
	--defaults beamer \
	-o $@
