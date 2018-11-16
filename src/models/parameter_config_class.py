class parameter_config():
	def __init__(self):
		self.num_input_features = 9 # Constant variable
		self.num_output_features = 3 # Constant variable
		self.num_decoder_features = 29 # Constant variable
		self.input_sequence_length = 28 # Constant variable
		self.target_sequence_length = 37 # Constant variable (order strict)
		self.target_list=['t2m','rh2m','w10m'] # Constant variable
		self.obs_range_dic={'t2m':[-30,42], # Official value: [-20,42]
		                 'rh2m':[0.0,100.0],
		                 'w10m':[0.0, 30.0]} # Constant variable
		self.obs_and_output_feature_index_map = {'t2m':0,'rh2m':1,'w10m':2} # Constant variable
		self.ruitu_feature_index_map = {'t2m':1,'rh2m':3,'w10m':4} # Constant variable
		#---------------Warning!--------------------
		# In this project, above parameters should not be changed!!
		# Because our data shape format only match these settings.
		# Changing them may cause the project not working.
		#---------------!---------------------

		self.num_steps_to_predict = 40 # This para has not been used yet. We will attend this function in the future.

		self.layers = [50,50]
		layers_str =''
		for i in self.layers:
			layers_str += str(i)+'_'

		self.lr = 0.001
		self.decay = 0
		self.loss = 'mae' # must be mve, mse, OR mae
		self.early_stop_limit = 10 # with the unit of Iteration Display
		self.pi_dic={0.95:1.96, 0.9:1.645, 0.8:1.28, 0.68:1.} # Gaussian distribution confidence interval ({confidence:variance})
		self.regulariser = None
		self.dropout_rate = 0

		self.id_embd = True
		self.time_embd = True
		self.model_name_format_str = '_layers_{}loss_{}_dropout{}'.\
									format(layers_str, self.loss, self.dropout_rate)
