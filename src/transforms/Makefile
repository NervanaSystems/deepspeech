.PHONY: default clean

default: libwarpctc.so

clean:
	@rm -rf warp-ctc/build
	@rm -rf libwarpctc.so

libwarpctc.so: warp-ctc/build/libwarpctc.so
	@ln -sf warp-ctc/build/libwarpctc.so libwarpctc.so

warp-ctc/build/libwarpctc.so:
	@rm -rf warp-ctc/build
	@mkdir warp-ctc/build
	@cd warp-ctc/build && cmake .. && make
	
