import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

def build_df(path,partition):
	df = pd.read_csv(path)
	df['Partition'] = [partition for x in range(0,len(df['X']))]
	return df

def find_limits(a,b):
	maximum = max([a['Throughput[GB/s]'].max(),b['Throughput[GB/s]'].max()])
	minimum = min([a['Throughput[GB/s]'].min(),b['Throughput[GB/s]'].min()])
	return maximum,minimum

def set_ax(df,ax,min,max):
	labels = [str(row['X'])+"x"+str(row['Y'])+"x"+str(row['Z']) for i,row in df.iterrows()]
	labels = labels[:13]
	for i,a in enumerate(ax):
		a.set_xlabel('Volume',fontsize=15)
		a.set_ylabel('Throughput [GB/s]',fontsize=15)
		a.set_xticks(range(0,13))
		a.set_xticklabels(labels,rotation=45,fontsize=7)
		a.set_ylim(min,max+2)
		#a.set_yscale('log')
		a.legend(loc=2,prop={'size': 12})
		if i == 0:
			a.set_title("Shared Memory",fontsize=20)
		else:
			a.set_title("Distributed Memory",fontsize=20)
		a.grid(True,which="both",ls="-")

def plot_df(df1,df2):
	global_max,global_min = find_limits(df1,df2)
	sns.set_style("white")
	fig, axes = plt.subplots(1,2,sharex=False,figsize=(15,8))
	sns.lineplot(ax=axes[0],data=df1,x=[x for x in range(0,13)]*2,y='Throughput[GB/s]',hue='Partition',marker='X',legend=True)
	sns.lineplot(ax=axes[1],data=df2,x=[x for x in range(0,13)]*2,y='Throughput[GB/s]',hue='Partition',marker='X',legend=True)
	set_ax(df1,axes,global_min,global_max)
	fig.savefig('stencil.pdf')
	fig.savefig('stencil.png', bbox_inches='tight')

if __name__ == "__main__":
#	Usage: Update the path to the CSV files accordingly.
#	Adjust the axes settings as needed (set_ax).
	skl_shared = 'stencil-shared-skl-upcxx.csv'
	knl_shared = 'stencil-shared-knl-upcxx.csv'
	skl_dist = 'stencil-dist-skl-upcxx.csv'
	knl_dist = 'stencil-dist-knl-upcxx.csv'
	skl = build_df(skl_shared,'Media')
	knl = build_df(knl_shared,'Knl')
	shared = pd.concat([skl,knl])
	skl = build_df(skl_dist,'Media')
	knl = build_df(knl_dist,'Knl')
	dist = pd.concat([skl,knl])
	plot_df(shared,dist)
