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

def set_ax(ax,min,max):
	labels = [ str(2**x)+"x"+str(2**y) for x,y in zip(range(5,15),range(5,15))]
	for i,a in enumerate(ax):
		a.set_xlabel('Matrix Dimension',fontsize=15)
		a.set_ylabel('Throughput [GB/s]',fontsize=15)
		a.set_xlim(4,15)
		a.set_xticks(range(5,15))
		a.set_xticklabels(labels,rotation=45,fontsize=7)
		a.set_ylim(min,max+500)
		a.set_yscale('log')
		a.legend(loc=2,prop={'size': 12})
		if i == 0:
			a.set_title("Shared Memory (OpenMP)",fontsize=20)
		else:
			a.set_title("Distributed Memory (OpenMP)",fontsize=20)
		a.grid(True,which="both",ls="-")

def plot_df(df1,df2):
	global_max,global_min = find_limits(df1,df2)
	sns.set_style("white")
	fig, axes = plt.subplots(1,2,sharex=False,figsize=(15,8))
	sns.lineplot(ax=axes[0],data=df1,x=[x for x in range(5,15)]*2,y='Throughput[GB/s]',hue='Partition',marker='X',legend=True)
	sns.lineplot(ax=axes[1],data=df2,x=[x for x in range(5,15)]*2,y='Throughput[GB/s]',hue='Partition',marker='X',legend=True)
	set_ax(axes,global_min,global_max)
	fig.savefig('symmetrize_openmp.pdf')
	fig.savefig('symmetrize_openmp.png', bbox_inches='tight')

if __name__ == "__main__":
#	Usage: Update the path to the CSV files accordingly.
#	Adjust the axes settings as needed (set_ax).
	skl_shared = 'csv/symmetrize-shared-skl-upcxx-openmp.csv'
	knl_shared = 'csv/symmetrize-shared-knl-upcxx-openmp.csv'
	skl_dist = 'csv/symmetrize-dist-skl-upcxx-openmp.csv'
	knl_dist = 'csv/symmetrize-dist-knl-upcxx-openmp.csv'

	skl = build_df(skl_shared,'Media')
	knl = build_df(knl_shared,'Knl')
	shared = pd.concat([skl,knl])
	skl = build_df(skl_dist,'Media')
	knl = build_df(knl_dist,'Knl')
	dist = pd.concat([skl,knl])
	plot_df(shared,dist)
