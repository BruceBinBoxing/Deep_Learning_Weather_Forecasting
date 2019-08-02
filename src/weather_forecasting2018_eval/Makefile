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


### Supplemental evaluation script in KDD paper. 
### This evaluation will show the evaluation for each of ten weather stations, as well as the totally "averaged evaluation".
### In the paper, the result is only the "averaged evaluation".
DEMO_MODEL_NAME := Seq2Seq_MVE_omitRUITU_layers_300_300__loss_mve_dropout0
### SS (skill score) evlaution
evaluate_0924_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018092403.csv --obs ./OBS_2018092403.csv --fore ./FORE_2018092403.csv
evaluate_1015_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018101503.csv --obs ./aic_wf2018_testa2_obs_2018101503.csv --fore ./aic_wf2018_testa2_fore_2018101503.csv
evaluate_1028_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018102803.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv
evaluate_1029_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018102903.csv --obs ./ai_challenger_wf2018_testb2_obs_2018102903.csv --fore ./ai_challenger_wf2018_testb2_fore_2018102903.csv
evaluate_1030_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018103003.csv --obs ./ai_challenger_wf2018_testb3_obs_2018103003.csv --fore ./ai_challenger_wf2018_testb3_fore_2018103003.csv
evaluate_1031_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018103103.csv --obs ./ai_challenger_wf2018_testb4_obs_2018103103.csv --fore ./ai_challenger_wf2018_testb4_fore_2018103103.csv
evaluate_1101_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018110103.csv --obs ./ai_challenger_wf2018_testb5_obs_2018110103.csv --fore ./ai_challenger_wf2018_testb5_fore_2018110103.csv
evaluate_1102_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018110203.csv --obs ./ai_challenger_wf2018_testb6_obs_2018110203.csv --fore ./ai_challenger_wf2018_testb6_fore_2018110203.csv
evaluate_1103_paper:
	python3 eval_details_score.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018110303.csv --obs ./ai_challenger_wf2018_testb7_obs_2018110303.csv --fore ./ai_challenger_wf2018_testb7_fore_2018110303.csv

### RMSE evlaution
evaluate_0924_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018092403.csv --obs ./OBS_2018092403.csv --fore ./FORE_2018092403.csv
evaluate_1015_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018101503.csv --obs ./aic_wf2018_testa2_obs_2018101503.csv --fore ./aic_wf2018_testa2_fore_2018101503.csv
evaluate_1028_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018102803.csv --obs ./ai_challenger_wf2018_testb1_obs_2018102803.csv --fore ./ai_challenger_wf2018_testb1_fore_2018102803.csv
evaluate_1029_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018102903.csv --obs ./ai_challenger_wf2018_testb2_obs_2018102903.csv --fore ./ai_challenger_wf2018_testb2_fore_2018102903.csv
evaluate_1030_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018103003.csv --obs ./ai_challenger_wf2018_testb3_obs_2018103003.csv --fore ./ai_challenger_wf2018_testb3_fore_2018103003.csv
evaluate_1031_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018103103.csv --obs ./ai_challenger_wf2018_testb4_obs_2018103103.csv --fore ./ai_challenger_wf2018_testb4_fore_2018103103.csv
evaluate_1101_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018110103.csv --obs ./ai_challenger_wf2018_testb5_obs_2018110103.csv --fore ./ai_challenger_wf2018_testb5_fore_2018110103.csv
evaluate_1102_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018110203.csv --obs ./ai_challenger_wf2018_testb6_obs_2018110203.csv --fore ./ai_challenger_wf2018_testb6_fore_2018110203.csv
evaluate_1103_rmse:
	python3 eval_details_rmse.py --submit ./paper_result_csv/$(DEMO_MODEL_NAME)-2018110303.csv --obs ./ai_challenger_wf2018_testb7_obs_2018110303.csv --fore ./ai_challenger_wf2018_testb7_fore_2018110303.csv
