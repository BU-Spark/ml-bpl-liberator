SHELL:=/bin/bash

run_model:
ifdef input_directory
	virtualenvs/venv1/bin/python first_model.py -i $(input_directory)
	virtualenvs/venv2/bin/python second_model.py
else
	virtualenvs/venv1/bin/python first_model.py
	virtualenvs/venv2/bin/python second_model.py
endif

run_model_scc:
ifdef input_directory
	module load python3/3.7.7 && virtualenvs/venv1/bin/python first_model.py -i $(input_directory)
	module load python3/3.8.3 && virtualenvs/venv2/bin/python second_model.py
else
	module load python3/3.7.7 && virtualenvs/venv1/bin/python first_model.py
	module load python3/3.8.3 && virtualenvs/venv2/bin/python second_model.py
endif

set_up_venvs:
	python3 -m virtualenv -p=$(python_path_one) virtualenvs/venv1 && source virtualenvs/venv1/bin/activate && pip install -r requirements_1.txt && deactivate
	python3 -m virtualenv -p=$(python_path_two) virtualenvs/venv2 && source virtualenvs/venv2/bin/activate && pip install -r requirements_2.txt && deactivate

set_up_venvs_scc:
	module load python3/3.8.3 && python3 -m virtualenv virtualenvs/venv2 && source virtualenvs/venv2/bin/activate && pip install -r requirements_2.txt && deactivate
	module load python3/3.7.7 && python3 -m virtualenv virtualenvs/venv1 && source virtualenvs/venv1/bin/activate && pip install -r requirements_1.txt && deactivate

clean:
	rm -rf data/npy_outputs/*
	rm -rf data/cropped_articles/*
	rm -rf data/output_images/*
	rm -rf data/segment_outputs/*