.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = weather_AI_Ch
PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	pip install -U pip setuptools wheel
	pip install -r requirements.txt

reproduce_experiments:

## Make Dataset
make_train_data:
	$(PYTHON_INTERPRETER) \
	src/data/make_TrainAndVal_Data_from_nc.py \
	'./data/raw/' \
	--process_phase train \
	'./data/interim/' \
	'./data/processed/'

make_val_data:
	$(PYTHON_INTERPRETER) \
	src/data/make_TrainAndVal_Data_from_nc.py \
	'./data/raw/' \
	--process_phase val \
	'./data/interim/'	\
	'./data/processed/'

# Please set --process_phase as your want (testA_as_val OR testB_as_val)
make_ValData_from_TestData:
	$(PYTHON_INTERPRETER) \
	src/data/make_ValData_from_TestData_from_nc.py \
	'./data/raw/' \
	--process_phase=testB_as_val \
	'./data/interim/'	\
	'./data/processed/'

make_TestOneDayOnlineData_for_submit:
	$(PYTHON_INTERPRETER) src/data/make_TestOnlineData_from_nc.py \
	'./data/raw/' \
	--process_phase OnlineEveryDay \
	'./data/interim/'	\
	--datetime 20181028 \
	'./data/processed/'

	#$(PYTHON_INTERPRETER) \
	src/data/make_TestOnlineData_from_nc.py \
	'./data/raw/' \
	--process_phase testB \
	'./data/interim/'	\
	--datetime 20181023 \
	'./data/processed/'

	#$(PYTHON_INTERPRETER) \
	src/data/make_TestOnlineData_from_nc.py \
	'./data/raw/' \
	--process_phase OnlineEveryDay \
	'./data/interim/' \
	--datetime 20181028 \
	'./data/processed/'

#MODEL_NAME := seq2seq_subnet_50_swish_dropout
MODEL_NAME := Seq2Seq_MVE
## Train and Predict
train_from_scratch:
	$(PYTHON_INTERPRETER) \
	src/run/Train_from_scratch.py \
	'./data/processed/' \
	--train_data='train_norm.dict' \
	--val_data='val_norm.dict' \
	./models/model_for_official/ \
	--model_name=$(MODEL_NAME)

MODEL_LIST=seq2seq_subnet_50_swish_dropout \
			seq2seq_subnet_30_30_best \
			seq2seq_subnet_200_200 \
			seq2seq_model_100 \
			seq2seq_subnet_50_50_dropout\
			seq2seq_model_250_250\
			seq2seq_subnet_100_swish_dropout

MODEL_LIST_NEW=seq2seq_model_best4937\
			seq2seq_subnet_200_200 \
			seq2seq_subnet_100_swish_dropout
# Please set 1. --test_file_name; 2. --saved_csv_path; 3.--saved_csv_name; depending on what you want.
load_multi_models_pred:
	for model in $(MODEL_LIST_NEW) ; do \
		echo $${model}.h5; \
    	$(PYTHON_INTERPRETER) src/run/Load_model_and_predict.py \
		'./models/' \
		--model_name $$model \
		'./data/processed/' \
		--test_file_name='OnlineEveryDay_20181028_norm.dict' \
		--saved_csv_path='./src/weather_forecasting2018_eval/ensemble_2018102803/' \
		--saved_csv_name=$$model-2018102803_demo.csv;\
    done

MODEL_NAME_prediction = Seq2Seq_MVE_layers_50_50_loss_mae_dropout0
load_single_model_and_predict:
	$(PYTHON_INTERPRETER) src/run/Load_model_and_predict.py \
	'./models/model_for_official/' \
	--model_name $(MODEL_NAME_prediction) \
	'./data/processed/' \
	--test_file_name='OnlineEveryDay_20181028_norm.dict' \
	--saved_csv_path='./src/weather_forecasting2018_eval/pred_result_csv/' \
	--saved_csv_name=$(MODEL_NAME_prediction)-2018102803_demo.csv

#make_dataset_from_netCDF2df:
	#$(PYTHON_INTERPRETER) src/data/make_dataset_from_nc2df.py '/Users/kudou/Documents/dataset/AI_challeng/' --process_phase train './data/interim/'
	#$(PYTHON_INTERPRETER) src/data/make_dataset_from_nc2df.py '/Users/kudou/Documents/dataset/AI_challeng/' --process_phase val './data/interim/'
	#$(PYTHON_INTERPRETER) src/data/make_dataset_from_nc2df.py '/Users/kudou/Documents/dataset/AI_challeng/' --process_phase='testA' './data/interim/'
	#$(PYTHON_INTERPRETER) src/data/make_dataset_from_nc2df.py '/Users/kudou/Documents/dataset/AI_challeng/' --process_phase='testB' './data/interim/'

#make_dataset_missing_fill_and_make_ndarray:
	#$(PYTHON_INTERPRETER) src/data/make_dataset_missing_fill.py './data/interim/' --obs_file_name=obs_df_train.pkl --ruitu_file_name=ruitu_df_train.pkl --station_id=999 './data/processed/'
	#$(PYTHON_INTERPRETER) src/data/make_dataset_missing_fill.py './data/interim/' --obs_file_name=obs_df_val.pkl --ruitu_file_name=ruitu_df_val.pkl --station_id=999 './data/processed/'
	#$(PYTHON_INTERPRETER) src/data/make_dataset_missing_fill.py './data/interim/' --obs_file_name=obs_df_test.pkl --ruitu_file_name=ruitu_df_test.pkl --station_id=999 './data/processed/'


data: requirements
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src

## Upload Data to S3
sync_data_to_s3:
ifeq (default,$(PROFILE))
	aws s3 sync data/ s3://$(BUCKET)/data/
else
	aws s3 sync data/ s3://$(BUCKET)/data/ --profile $(PROFILE)
endif

## Download Data from S3
sync_data_from_s3:
ifeq (default,$(PROFILE))
	aws s3 sync s3://$(BUCKET)/data/ data/
else
	aws s3 sync s3://$(BUCKET)/data/ data/ --profile $(PROFILE)
endif

## Set up python interpreter environment
create_environment:
ifeq (True,$(HAS_CONDA))
		@echo ">>> Detected conda, creating conda environment."
ifeq (3,$(findstring 3,$(PYTHON_INTERPRETER)))
	conda create --name $(PROJECT_NAME) python=3
else
	conda create --name $(PROJECT_NAME) python=2.7
endif
		@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@pip install -q virtualenv virtualenvwrapper
	@echo ">>> Installing virtualenvwrapper if not already intalled.\nMake sure the following lines are in shell startup file\n\
	export WORKON_HOME=$$HOME/.virtualenvs\nexport PROJECT_HOME=$$HOME/Devel\nsource /usr/local/bin/virtualenvwrapper.sh\n"
	@bash -c "source `which virtualenvwrapper.sh`;mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER)"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
endif

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
