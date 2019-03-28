import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import brewer2mpl
import matplotlib as mpl


def visualize_log(filename, figsize=None, output=None):
	with open(filename, 'r') as f:
		data = json.load(f)
	if 'episode' not in data:
		raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
	episodes = data['episode']

	# Get value keys. The x axis is shared and is the number of episodes.
	keys = sorted(list(set(data.keys()).difference(set(['episode']))))

	if figsize is None:
		figsize = (15., 5. * len(keys))
	#f, axarr = plt.subplots(len(keys), sharex=True, figsize=figsize)#vs
	'''
	for i in range(len(data['episode_reward'])):

		if data['episode_reward'][i] < -4600:
			data['episode_reward'][i] = data['episode_reward'][i-1]
			#data['episode_reward'].remove(data['episode_reward'][i])
			#episodes.remove(episodes[i])
		if i>1500 and data['episode_reward'][i] < -1400:
			data['episode_reward'][i] = data['episode_reward'][i-1]
		if i==0  and data['episode_reward'][i] > -1400:
			data['episode_reward'][i] = data['episode_reward'][i+5]
		if i<25:
			data['loss'][i] = data['loss'][i+25]
	'''
	episode_reward  = [x/-3911.2734714782787 for x in data['episode_reward']]
	'''
	for i in range(len(episode_reward)):
		if episode_reward[i] > 15:
			#print("hii")
			episode_reward[i] = episode_reward[i-1]
	'''
	'''
	for i in range(0,10):
		data['loss'][i] = data['loss'][i+10]
	for i in range(len(data['episode_reward'])):

		if data['episode_reward'][i] > 3:
			data['episode_reward'][i] = data['episode_reward'][i-1]
	'''
	'''
	for idx, key in enumerate(keys):

		#print(len(episodes), len(data["episode_reward"]))
		newList = [x  for x in data[key]]
		axarr[idx].plot(episodes, newList)
		axarr[idx].set_ylabel(key)
	'''
	# DDPG plot
	darray = [79,89,89,90,91,94,98,99,104,103,119,110,108,110,115,115,117,121,123,123,128,131,137,136,138,140,138,141,145,155,156,\
				155,160,160,170,165,174,167,166,166,177,180,181,181,193,200,199,192,200,195,197,200,196,192,198,212,228,231,\
				241,221,235,220,233,232,227,230,232,254,251,243,257,251,264,264,255,241,244,242,241,247,251,256,262,260,\
				258,264,264]

	#[82,87,90,93,100,103,93,101,100,106,109,110,114,112,114,121,121,123,125,125,128,126,141,135,144,141,151,158,142,153,166,\
	#			167,163,155,162,158,162,164,168,164,167,174,180,187,182,173,180,178,188,183,183,192,198,205,202,197,189,193,187,193,196,\
	#			214,209,197,200,205,205,210,223,216,213,218,223,242,232,226,234,227,240,239]

	#74,80,86,83,86,87,89,92,91,95,98,98,100,104,106,108,109,111,114,119,119,123,128,123,126,126,132,131,141,141,142,144,158,152,150,153,154,158\
	#			,162,163,163,177,178,174,171,177,175,179,196,193,189,208,210,197,193,195,209,210,217,221,213,217,219,221,225,221,223,223,228,236,231,234\
	#			,239,251,267,291,295,289,311,309,338,299,316,317,309,309,270,273,269,290,287,285,303,292,314,304,291,297,297,305,309,307,310,308,306,320,\
	#			420,326,338,347,360,361,362,386,397,418,467,464,427,347,349,349,355,351,356,363,364,371,374,372,375,381,449,479,493,462,415,403,401,510,\
	#			470,449,427,427,436,451,450,461,460,449,467,522,637,550,511,516,529,523,487,493,500,539,556,553,555,566,569,561,549,550,553,558,564,571,575,\
	#			579,583,585,591,593,603,609,612,620,627,629,644,641,647,651,659,666,673,672,675,683,690,741,756,761,764,758,731,731,747,745,762,769,763,768,\
	#			773,817,825,792,797,835,873,861,973,878]
	episodes_per_echo = 11
	t=0
	tarray = np.arange(t, t + darray[1], (darray[1])/episodes_per_echo)
	for d in range(0, len(darray)-1):
		t += darray[d]
		if d<7:	#51
			ind = np.arange(t, t + darray[d+1], (darray[d+1])/10)
		else:
			ind = np.arange(t, t + darray[d+1], (darray[d+1])/11)
		tarray = np.concatenate((tarray, ind),axis=None)

	print(len(tarray), len(episodes),tarray)


	#plot preprocessing
	bmap = brewer2mpl.get_map('Set2','qualitative', 7)
	colors = bmap.mpl_colors

	params = {
	'axes.labelsize': 10,
	'font.size': 8,
    'legend.fontsize': 15,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'text.usetex': True	,
    'figure.figsize': [8, 6], # instead of 4.5, 4.5
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'ps.useafm' : True,
    'pdf.use14corefonts':True
    #'pdf.fonttype': 42,
    #'ps.fonttype': 42
	 }
	mpl.rcParams.update(params)


	b, a = signal.butter(8  , 0.025)
	plt.plot(tarray, episode_reward, color=colors[1], alpha=0.9)
	plt.plot(tarray, signal.filtfilt(b, a, episode_reward), color=colors[2], linewidth=3)
	plt.grid(axis='y', color='.910', linestyle='-', linewidth=1.5)
	plt.grid(axis='x', color='.910', linestyle='-', linewidth=1.5)

	plt.xlabel('Training time (seconds)', fontsize=16)
	plt.ylabel('Episodic reward fraction', fontsize=16)
	plt.legend(['Original','Filtered'])
	
  
	plt.tight_layout()

	if output is None:
		plt.show()
	else:
		plt.savefig(output)


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='The filename of the JSON log generated during training.')
parser.add_argument('--output', type=str, default=None, help='The output file. If not specified, the log will only be displayed.')
parser.add_argument('--figsize', nargs=2, type=float, default=None, help='The size of the figure in `width height` format specified in points.')
args = parser.parse_args()

# You can use visualize_log to easily view the stats that were recorded during training. Simply
# provide the filename of the `FileLogger` that was used in `FileLogger`.
visualize_log(args.filename, output=args.output, figsize=args.figsize)
