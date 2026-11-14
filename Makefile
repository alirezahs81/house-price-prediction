.PHONY: install test eda train predict
install:
	python -m pip install -r requirements.txt
test:
	python -m pytest
eda:
	python -m main eda
train:
	python -m main train
predict:
	python -m main predict
