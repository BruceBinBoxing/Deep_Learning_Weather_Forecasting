evaluate_a1_submit1:
	python3 weather_forecasting2018_eval.py --submit ./forecast-2018092403_submit1.csv --obs ./OBS_2018092403.csv --fore ./FORE_2018092403.csv
evaluate_a1_best_submit2:
	python3 weather_forecasting2018_eval.py --submit ./forecast-2018092403_submit2.csv --obs ./OBS_2018092403.csv --fore ./FORE_2018092403.csv
evaluate_a1_submit3:
	python3 weather_forecasting2018_eval.py --submit ./forecast-2018092403_submit3.csv --obs ./OBS_2018092403.csv --fore ./FORE_2018092403.csv


#MODEL_NAME := seq2seq_subnet_50_swish_dropout
DEMO_MODEL_NAME := Seq2Seq_MVE_layers_50_50_loss_mae_dropout0

evaluate_offline_0924_seq2seq:
	python3 weather_forecasting2018_eval_my.py --submit ./pred_result_csv/$(DEMO_MODEL_NAME)-2018092403.csv --obs ./OBS_2018092403.csv --fore ./FORE_2018092403.csv
evaluate_offline_1015_seq2seq:
	python3 weather_forecasting2018_eval_my.py --submit ./pred_result_csv/$(DEMO_MODEL_NAME)-2018101503.csv --obs ./aic_wf2018_testa2_obs_2018101503.csv --fore ./aic_wf2018_testa2_fore_2018101503.csv

evaluate_ensemble_1015:
	python3 weather_forecasting2018_eval_my.py --submit ./ensemble_2018101503/ensemble_avg.csv --obs ./aic_wf2018_testa2_obs_2018101503.csv --fore ./aic_wf2018_testa2_fore_2018101503.csv
evaluate_ensemble_0924:
	python3 weather_forecasting2018_eval_my.py --submit ./ensemble_2018092403/ensemble_avg.csv --obs ./OBS_2018092403.csv --fore ./FORE_2018092403.csv

evaluate_1028_demo:
	python3 weather_forecasting2018_eval.py --submit ./pred_result_csv/Seq2Seq_MVE_layers_50_50_loss_mae_dropout0-2018102803.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv
evaluate_1028_demo_ensemble:
	python3 weather_forecasting2018_eval.py --submit ./ensemble_2018102803/ensemble_avg.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv


evaluate_1028_1:
	python3 weather_forecasting2018_eval_my.py --submit ./pred_result_csv/submit_csv/forecast-2018102803-submit1.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv
evaluate_1028_2:
	python3 weather_forecasting2018_eval_my.py --submit ./pred_result_csv/submit_csv/forecast-2018102803-submit2.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv
evaluate_1028_3:
	python3 weather_forecasting2018_eval_my.py --submit ./pred_result_csv/submit_csv/forecast-2018102803-submit3.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv
evaluate_1028_s1:
	python3 weather_forecasting2018_eval_my.py --submit ./ensemble_2018102803/seq2seq_subnet_100_swish_dropout-2018102803.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv
evaluate_1028_lhs:
	python3 weather_forecasting2018_eval_my.py --submit ./ensemble_2018102803_2/ensemble_avg.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv
