.PHONY: default all clean

default: all

all: src/transforms/libwarpctc.so

clean:
	@$(MAKE) -C src/transforms clean
	@find . -name '*.pyc' -delete

src/transforms/libwarpctc.so:
	@$(MAKE) -C src/transforms


