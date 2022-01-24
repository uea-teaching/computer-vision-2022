# just type make to compile

build:
	docker run --rm -it -v "${PWD}"/lectures:/src dgrnwd/teachingslides:latest
	docker run --rm -it -v "${PWD}"/labs:/src dgrnwd/teachingslides:latest
