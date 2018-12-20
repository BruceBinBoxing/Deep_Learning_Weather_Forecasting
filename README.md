Global AI Challenger-Weather Forecasting
==============================
> Using deep learning produces more accurate weather forecasts based on the observation and prediction of meteorological elements.

Our rank-3 (CCIT007) score indicates end-to-end seq2seq Deep Learning model with small feature engineering can greatly improve weather forecasting accuracy (This is our first competition project. If it is helpful, a star would be a big stimuli for us. Many thanks to *Huaishao Luo's* big help.)
### License
Apache
### Project Framework
I feel [cookiecutter-data-science](https://drivendata.github.io/cookiecutter-data-science/) is helpful to organize project codes. Hope you also like it.
### Requirements
I test it on **MacOS** and **Ubuntu**. It is based on **Python 3.6.** Required packages like keras, tensorflow etc. are iincluded in **requirements.txt**. Run bellow command to install them.
> pip install -r requirements.txt
### Pipeline for quick start.
Go to https://challenger.ai/competition/wf2018. to downlowd 3 datasets as bellow (You can switch to English from top-right corner):

Training set: **ai_challenger_wf2018_trainingset_20150301-20180531.nc**  
Validation set: **ai_challenger_wf2018_validation_20180601-20180828_20180905.nc**  
Test set (Taking one-day test data on 28/10/2018 as an example): **ai_challenger_wf2018_testb1_20180829-20181028**

After downloaded, set three original dataset into the folder ./data/raw/
(For quick start, we just take  **ai_challenger_wf2018_testb1_20180829-20181028** as a test example, you can easily re-name the related arguments and apply it on other test set, which can be introduced in later section **How to change test dataset for other days?** )

Ensure that you are located in the project root directory and implement commands **strictly** according to bellow order:

  1. *make make_train_data* **(Prepare training data)**
  2. *make make_val_data* **(Prepare validation data)**
  3. *make make_TestOneDayOnlineData_for_submit* **(Prepare one-day test data)**
  4. *make train_from_scratch* **(Train)**
  5. *make load_single_model_and_predict* **(Test)**

      - Then a file called **'Seq2Seq_MVE_layers_50_50_loss_mae_dropout0-2018102803_demo.csv'** is created in the path './src/weather_forecasting2018_eval/pred_result_csv/'.
      To evaluate, locate yourself to the path'./src/weather_forecasting2018_eval' and run:
  6. *make evaluate_1028_demo*

      - You will get the resulted score (This day i.e., 28/10 is relatively hard to predict, we only got score of 0.2408 using this demo model). BTW, to reproduce our online scores, you can evaluate all submitted files in  'src/weather_forecasting2018_eval/pred_result_csv/submit_csv' by changing evaluated file name.
      - Since ensemble can improve the prediction accuracy and stability, I have trained many models for you to try ensemble! Go back to the the project root directory and run next command 7:

  7. *make load_multi_models_pred*

      - Three different models are loaded and predict in turn. The resulted .csv file of each model is saved in './src/weather_forecasting2018_eval/ensemble_2018102803/'
      - Then locate yourself to ./src/weather_forecasting2018_eval/ensemble_2018102803/ and run next command 8:

  8. *python ensemble.py*

      - Here ensemble is to simply calculate mathematical mean. You will get the file called ensemble_avg_2018102803.csv. Locate yourself to ./src/weather_forecasting2018_eval and run:
  9. *make evaluate_1028_demo_ensemble*

      - You will find the score of ensemble. Maybe sometimes you can find that ensemble result is lower than single model. Don't be confused about this. The performance of a single model can fluctuate very much according to different dataset. However stability is a great trait of ensemble learning. We  got score of 0.3440 using this demo ensemble model, which is a little higher than our online score 0.3358.

### How to change test dataset for other days?
Everyday, we have three opportunities to submit. We will use 3 models

**1. Ensemble model.**

  During the first competition days: we use models for ensemble including:

	MODEL_LIST=seq2seq_subnet_50_swish_dropout \
			seq2seq_subnet_30_30_best \
			seq2seq_subnet_200_200 \
			seq2seq_model_100 \
			seq2seq_subnet_50_50_dropout\
			seq2seq_model_250_250\
			seq2seq_subnet_100_swish_dropout

  During the last competition days: we try switching to models:

    MODEL_LIST_NEW=seq2seq_model_best4937\
    			seq2seq_subnet_200_200 \
    			seq2seq_subnet_100_swish_dropout

**2. Single model 1:** Seq2Seq_MVE_layers_222_222_loss_mve_dropout0. (At ./models/model_for_official)

**3. Single model 2:** Seq2Seq_MVE_layers_50_50_loss_mae_dropout0 (At ./models/model_for_official)
#### Pipeline steps:

  1. **Download raw test data oneline to the path: ./data/raw/**
  2. **Edit file** '/src/data/make_TestOnlineData_from_nc.py'. Set *file_name* to the same of the downloaded file in bellow snippet:
  > elif process_phase == 'OnlineEveryDay':
      file_name='ai_challenger_wf2018_testb1_20180829-20181028.nc'
  3. **Modify Makefile file for the rule 'make_TestOneDayOnlineData_for_submit'**:

    - 3.1 Set --process_phase=OnlineEveryDay
    - 3.2 Set --datetime=2018xxxx to month and day of prediction.
    - 3.3 Run 'make make_TestOneDayOnlineData_for_submit'

  4. **Modify Makefile for rule 'load_multi_models_pred'**: (Take datetime=2018xxxx as instance)

    - 4.1 Set --test_file_name = 'OnlineEveryDay_2018xxxx_norm.dict'
    - 4.2 Set --saved_csv_path = './src/weather_forecasting2018_eval/ensemble_2018xxxx03/'
    - 4.3 Set --saved_csv_name=$$model-2018xxxx03.csv
    - 4.4 Run 'make load_multi_models_pred'
    - 4.5 Locate to './src/weather_forecasting2018_eval/ensemble_2018xxxx03/', run 'python ensemble.py' and create 'ensemble_avg.csv' file. Copy and rename it to 'forecast-2018xxxx03.csv' then SUBMIT (1) it.

  5. **Modify Makefile for rule 'load_single_model_and_predict'**

    - 5.1 set MODEL_NAME_prediction = Seq2Seq_MVE_layers_50_50_loss_mae_dropout0 (Or another model you have trained)
    - 5.2 set --test_file_name='OnlineEveryDay_2018xxxx_norm.dict' \
    - 5.3 set --saved_csv_name=$(MODEL_NAME_prediction)-2018xxxx03.csv
    - 5.4 run 'make load_single_model_and_predict'
    - 5.5 set MODEL_NAME_prediction = Seq2Seq_MVE_layers_222_222_loss_mve_dropout0
    - 5.6 run 'make load_single_model_and_predict'

  6. **Locate to './src/weather_forecasting2018_eval/pred_result_csv/'**:

    - 6.1 Rename Seq2Seq_MVE_layers_50_50_loss_mae_dropout0-2018xxxx03.csv to 'forecast-2018xxxx03.csv' and SUBMIT (2)
    - 6.2 Rename Seq2Seq_MVE_layers_222_222_loss_mve_dropout0-2018xxxx03.csv to 'forecast-2018xxxx03.csv' and SUBMIT (3)

  7. Sanity check by visualization (not try).

     - We'd better plot for Sanity check before submitting! (in notebook or Excel). But we do not try this.
     
### Customize model parameters

You can dive into the './src/models/parameter_config_class.py'. Due to the parameters of the deep model are too many. Here we do not play exhaustedly. We mainly play different parameters of 'self.layers' and use ensemble to combine shallow and deep seq2seq model.

Hope your like this project. Please let me know your ideas and questions.


Best regards.

*Bin*

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
