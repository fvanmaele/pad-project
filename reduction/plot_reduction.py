import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

def build_df(path,partition):
        df = pd.read_csv(path)
        df['Partition'] = [partition for x in range(0,len(df['Size']))]
        return df

def find_limits(a,b):
        maximum = max([a['Throughput[GB/s]'].max(),b['Throughput[GB/s]'].max()])
        minimum = min([a['Throughput[GB/s]'].min(),b['Throughput[GB/s]'].min()])
        return maximum,minimum

def set_ax(ax,min,max):
        for i,a in enumerate(ax):
                a.set_xlabel('#Elements in powers of two',fontsize=15)
                a.set_ylabel('Throughput [GB/s]',fontsize=15)
                a.set_xlim(14,31)
                a.set_xticks(np.arange(15,31))
                a.set_ylim(min,max+500)
                a.set_yscale('log')
                a.legend(loc=2,prop={'size': 12})
                if i == 0:
                        a.set_title("Shared Memory",fontsize=20)
                else:
                        a.set_title("Distributed Memory",fontsize=20)
                a.grid(True,which="both",ls="-")

def plot_df(df1,df2):
        global_max,global_min = find_limits(df1,df2)
        sns.set_style("white")
        fig, axes = plt.subplots(1,2,sharex=True,figsize=(15,7))
        sns.lineplot(ax=axes[0],data=df1,x='Size',y='Throughput[GB/s]',hue='Partition',marker='X',legend=True)
        sns.lineplot(ax=axes[1],data=df2,x='Size',y='Throughput[GB/s]',hue='Partition',marker='X',legend=True)
        set_ax(axes,global_min,global_max)
        fig.savefig('reduction.pdf')

if __name__ == "__main__":
#       Usage: Update the path to the CSV files accordingly.
#       Adjust the axes settings as needed (set_ax).
        skl_shared = 'csv/reduction-skl-shared-upcxx-8.csv'
        knl_shared = 'csv/reduction-knl-shared-upcxx-16.csv'
        skl_dist = 'csv/reduction-skl-dist-upcxx-16.csv'
        knl_dist = 'csv/reduction-knl-dist-upcxx-16.csv'

        #skl_serial = 'csv/reduction-skl-serial.csv'
        #knl_serial = 'csv/reduction-knl-serial.csv'
        #skl_shared_openmp = 'csv/reduction-skl-shared-openmp.csv'
        #knl_shared_openmp = 'csv/reduction-knl-shared-openmp.csv'

        skl = build_df(skl_shared,'Media')
        knl = build_df(knl_shared,'Knl')
        shared = pd.concat([skl,knl])
        skl = build_df(skl_dist,'Media')
        knl = build_df(knl_dist,'Knl')
        dist = pd.concat([skl,knl])
        plot_df(shared,dist)
