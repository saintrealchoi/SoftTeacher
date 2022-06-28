pre:
	python -m pip install -r requirements.txt 
	mkdir -p thirdparty
	git clone https://github.com/saintrealchoi/mmdetection.git thirdparty/mmdetection
	cd thirdparty/mmdetection && python -m pip install -e .
install:
	make pre
	python -m pip install -e .
clean:
	rm -rf thirdparty
	rm -r ssod.egg-info
