run_model:
ifdef input_directory
	rm -rf data/npy_outputs/*
	rm -rf data/cropped_articles/*
	rm -rf data/output_images/*
	rm -rf data/segment_outputs/*
	virtualenvs/venv1/bin/python first_model.py -i $(input_directory)
	virtualenvs/venv2/bin/python second_model.py
else
	rm -rf data/npy_outputs/*
	rm -rf data/cropped_articles/*
	rm -rf data/output_images/*
	rm -rf data/segment_outputs/*
	virtualenvs/venv1/bin/python first_model.py
	virtualenvs/venv2/bin/python second_model.py
endif