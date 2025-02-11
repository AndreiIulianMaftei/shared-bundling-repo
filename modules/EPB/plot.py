import copy
import os
import networkx as nx
import numpy as np
from scipy import signal
from PIL import Image
import pickle
import seaborn as sbn
import matplotlib.pyplot as plt
import matplotlib
from frechetdist import frdist
import re
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from pdf2image import convert_from_path
import requests
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_violin, geom_boxplot, theme, element_text, labs, element_blank

class Plot: 
    def __init__(self):
        self = self

    def adjacent_values(vals, q1, q3):
            upper_adjacent_value = q3 + (q3 - q1) * 1.5
            upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1]).item()

            lower_adjacent_value = q1 - (q3 - q1) * 1.5
            lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1).item()
            return lower_adjacent_value, upper_adjacent_value

    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Violin')

    def plotMonotonicity(rezults):
        Max = -999999999999
        Min = 999999999999
        mean = [0,0,0,0]
        number = 0
        nr = 1
        print(mean)
        for rez in rezults: 
            if(rez[1] == "epb"):
                Max = Max
                Min = Min
                continue
            Max = max(np.max(rez[0]), Max)
            Min = min(np.min(rez[0]), Min)
            mean[0] += np.sum(rez[0])
            mean[nr] += np.sum(rez[0])
            mean[nr] = mean[nr]/rez[0].__len__()
            number = number+ rez[0].__len__()
        mean[0] = mean[0]/number
            
        print (rezults.__len__())
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        list_normalised_min_max = [None]*rezults.__len__()
        list_normalised_z_score = [None]*rezults.__len__()
        list = [None]*rezults.__len__()
        for index, rez in enumerate(rezults):
            if(rez[1] == "epb"):
                list_normalised_min_max[index] = []
                continue
            #normalise rez[0] (a list ) using min/max
            list[index] = rez[0]
            #print(rez[0])
            list_normalised_min_max[index] = [(x - Min) / (Max - Min) for x in [list[index]]]
            #list_normalised_z_score[index] = [(x - mean) / np.std(list[index]) for x in list[index]]
        #print single histogram
        png_file_min_max = []
        png_file_z_score = []
        for index, rez in enumerate(rezults):
            plt.figure()
            plt.hist(list_normalised_min_max[index], bins=100, alpha=0.5, label=f'Algorithm {rez[1]} MEAN : {mean[index+1]}')
            plt.legend(loc='upper right')
            plt.xlabel('Frechet Distance')
            plt.ylabel('Number of Edges')
            plt.savefig(f'monotonicity_normalisation_{rez[1]}.png')
            png_file_min_max.append(f'min-max normalisation {rez[1]}.png')
            plt.figure()
        

        # === Your Normalized Data (Replace with your actual data) ===
        list_normalised_min_max = [
            np.sort(np.random.rand(np.random.randint(80, 120))),
            np.sort(np.random.rand(np.random.randint(80, 120))),
            np.sort(np.random.rand(np.random.randint(80, 120)))
        ]
        labels = ['Sample A', 'Sample B', 'Sample C']
        # ============================================================

        # Calculate quartiles individually
        quartile1 = []
        medians = []
        quartile3 = []

        for dataset in list_normalised_min_max:
            q1, med, q3 = np.percentile(dataset, [25, 50, 75])
            quartile1.append(float(q1))
            medians.append(float(med))
            quartile3.append(float(q3))

        # Calculate whiskers individually
        whiskers_min = []
        whiskers_max = []

        for sorted_array, q1, q3 in zip(list_normalised_min_max, quartile1, quartile3):
            whisker_min, whisker_max = Plot.adjacent_values(sorted_array, q1, q3)
            whiskers_min.append(whisker_min)
            whiskers_max.append(whisker_max)

        # Convert lists to NumPy arrays
        quartile1 = np.array(quartile1)
        medians = np.array(medians)
        quartile3 = np.array(quartile3)
        whiskers_min = np.array(whiskers_min)
        whiskers_max = np.array(whiskers_max)

        # Create the plot
        fig, ax = plt.subplots(figsize=(9, 4))

        parts = ax.violinplot(
            list_normalised_min_max, showmeans=False, showmedians=False, showextrema=False
        )

        # Customize the appearance
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')  # Violin fill color
            pc.set_edgecolor('black')    # Violin edge color
            pc.set_alpha(1)

        # Add scatter, quartile, and whisker lines
        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        # Set labels and title
        Plot.set_axis_style(ax, labels)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Violin Plot of Normalized Data')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig('violin_plot_monotonicity.png', dpi=300)


    def plotFrechet( rezults):
        '''
        Plot the normalised Frechet distance of two algorithms. a histogram, withn x axis as the frechet distance and y axis as the number of edges .
        '''
        Max = -999999999999
        Min = 999999999999
        mean = [0,0,0,0]
        
        number = 0
        nr = 1
        for rez in rezults: 
            Max = max(np.max(rez[0]), Max)
            Min = min(np.min(rez[0]), Min)
            mean[0] += np.sum(rez[0])
            mean[nr] += np.sum(rez[0])
            mean[nr] = mean[nr]/rez[0].__len__()
            nr += 1

            number = number+ rez[0].__len__()
        mean[0] = mean[0]/number
        
            
        print (rezults.__len__())
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        list_normalised_min_max = [None]*rezults.__len__()
        list = [None]*rezults.__len__()
        for index, rez in enumerate(rezults):
            list[index] = rez[0]
            
            list_normalised_min_max[index] = [(x - Min) / (Max - Min) for x in list[index]]
            mean[index+1] = np.mean(list_normalised_min_max[index])
        
        png_file_min_max = []
        for index, rez in enumerate(rezults):
            plt.figure()
            plt.xlim(-0.2, 1.2)
            plt.ylim(0, 80)
            plt.hist(list_normalised_min_max[index], bins=100, alpha=0.5, label=f'Algorithm {rez[1]} MEAN : {mean[index+1]}')
            plt.legend(loc='upper right')
            plt.xlabel('Frechet Distance')
            plt.ylabel('Number of Edges')
            plt.savefig(f'frechet_normalisation_{rez[1]}.png')
            png_file_min_max.append(f'frechet_normalisation_{rez[1]}.png')
            plt.figure()
            
        plt.figure()
        for index, rez in enumerate(rezults):
            plt.hist(list_normalised_min_max[index], bins=100, alpha=0.5, label=f'Algorithm {rez[1]}')
            plt.legend(loc='upper right')
            plt.xlabel('Frechet Distance')
            plt.ylabel('Number of Edges')    
        
        plt.savefig(f'min-max normalisation cumulative.png')
        
        quartile1 = []
        medians = []
        quartile3 = []

        for dataset in list_normalised_min_max:
            q1, med, q3 = np.percentile(dataset, [25, 50, 75])
            quartile1.append(q1)
            medians.append(med)
            quartile3.append(q3)
        whiskers = np.array([
            Plot.adjacent_values(sample, q1, q3)
            for sample, q1, q3 in zip(list_normalised_min_max, quartile1, quartile3)
        ])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        # Create the plot
        fig, ax = plt.subplots(figsize=(9, 4))

        parts = ax.violinplot(
            list_normalised_min_max, showmeans=False, showmedians=False, showextrema=False
        )

        # Customize the appearance
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')  # Violin fill color
            pc.set_edgecolor('black')    # Violin edge color
            pc.set_alpha(1)

        # Add scatter, quartile, and whisker lines
        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        # Set labels and title
        labels = ['FD', 'EPB', 'WR' ]  # Customize as needed
        Plot.set_axis_style(ax, labels)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Violin Plot of Normalized Data')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig('violin_plot_frechet.png', dpi=300)


    def plotDistortionHistogram(self, rezults):
        
        Max = -999999999999
        Min = 999999999999
        mean = [0,0,0,0]
        number = 0
        nr = 1
        for rez in rezults: 
            Max = max(np.max(rez[0]), Max)
            Min = min(np.min(rez[0]), Min)
            mean[0] += np.sum(rez[0])
            mean[nr] += np.sum(rez[0])
            mean[nr] = mean[nr]/rez[0].__len__()
            number = number+ rez[0].__len__()
        mean[0] = mean[0]/number
            
        print (rezults.__len__())
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        list_normalised_min_max = [None]*rezults.__len__()
        list_normalised_z_score = [None]*rezults.__len__()
        list = [None]*rezults.__len__()
        for index, rez in enumerate(rezults):
            #normalise rez[0] (a list ) using min/max
            list[index] = rez[0]
            #print(rez[0])
            list_normalised_min_max[index] = [(x - Min) / (Max - Min) for x in [list[index]]]
            #list_normalised_z_score[index] = [(x - mean) / np.std(list[index]) for x in list[index]]
        #print single histogram
        png_file_min_max = []
        png_file_z_score = []
        for index, rez in enumerate(rezults):
            plt.figure()
            plt.hist(list_normalised_min_max[index], bins=100, alpha=0.5, label=f'Algorithm {rez[1]} MEAN : {mean[index+1]}')
            plt.legend(loc='upper right')
            plt.xlabel('Frechet Distance')
            plt.ylabel('Number of Edges')
            plt.savefig(f'distortion_normalisation_{rez[1]}.png')
            png_file_min_max.append(f'min-max normalisation {rez[1]}.png')
            plt.figure()
        

        # === Your Normalized Data (Replace with your actual data) ===
        list_normalised_min_max = [
            np.sort(np.random.rand(np.random.randint(80, 120))),
            np.sort(np.random.rand(np.random.randint(80, 120))),
            np.sort(np.random.rand(np.random.randint(80, 120)))
        ]
        labels = ['Sample A', 'Sample B', 'Sample C']
        # ============================================================

        # Calculate quartiles individually
        quartile1 = []
        medians = []
        quartile3 = []

        for dataset in list_normalised_min_max:
            q1, med, q3 = np.percentile(dataset, [25, 50, 75])
            quartile1.append(float(q1))
            medians.append(float(med))
            quartile3.append(float(q3))

        # Calculate whiskers individually
        whiskers_min = []
        whiskers_max = []

        for sorted_array, q1, q3 in zip(list_normalised_min_max, quartile1, quartile3):
            whisker_min, whisker_max = Plot.adjacent_values(sorted_array, q1, q3)
            whiskers_min.append(whisker_min)
            whiskers_max.append(whisker_max)

        # Convert lists to NumPy arrays
        quartile1 = np.array(quartile1)
        medians = np.array(medians)
        quartile3 = np.array(quartile3)
        whiskers_min = np.array(whiskers_min)
        whiskers_max = np.array(whiskers_max)

        # Create the plot
        fig, ax = plt.subplots(figsize=(9, 4))

        parts = ax.violinplot(
            list_normalised_min_max, showmeans=False, showmedians=False, showextrema=False
        )

        # Customize the appearance
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')  # Violin fill color
            pc.set_edgecolor('black')    # Violin edge color
            pc.set_alpha(1)

        # Add scatter, quartile, and whisker lines
        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        # Set labels and title
        Plot.set_axis_style(ax, labels)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Violin Plot of Normalized Data')

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig('violin_plot_distortion.png', dpi=300)

        # Display the plot