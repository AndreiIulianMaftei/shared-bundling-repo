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
from wand.image import Image
import pandas as pd
from plotnine import ggplot, aes, geom_violin, geom_boxplot, theme, element_text, labs, element_blank
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.legend import Legend


class PlotTest:
    def __init__(self):
        pass

    def plotMonotonicity(self, results):
        self.plot_metrics(results, 'Monotonicity')

    def plotFrechet(self, results):
        self.plot_metrics(results, 'Frechet')

    def plotDistortionHistogram(self, results):
        self.plot_metrics(results, 'Distortion')

    def plotProjectedMonotonicity(self, results):
        self.plot_metrics(results, 'Monotonicity Projection')
    
    def plotAngles(self, results):
        self.plot_metrics(results, 'Angle')

    @staticmethod
    def adjacent_values(vals, q1, q3):
        upper_adjacent_value = q3 + (q3 - q1) * 1.5
        upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1]).item()

        lower_adjacent_value = q1 - (q3 - q1) * 1.5
        lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1).item()
        return lower_adjacent_value, upper_adjacent_value

    @staticmethod
    def set_axis_style(ax, labels):
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Algorithm')

    @staticmethod
    def normalize_data(data_list, method='min-max'):
        """
        Normalize the data using specified method.

        Parameters:
            data_list (list of lists): The data to normalize.
            method (str): 'min-max' or 'z-score'

        Returns:
            list of lists: Normalized data.
        """
        normalized_data = []
        if method == 'min-max':
            all_data = np.concatenate(data_list)
            min_val = np.min(all_data)
            max_val = np.max(all_data)
            for data in data_list:
                normalized = (data - min_val) / (max_val - min_val)
                normalized_data.append(normalized)
        elif method == 'z-score':
            all_data = np.concatenate(data_list)
            mean_val = np.mean(all_data)
            std_val = np.std(all_data)
            for data in data_list:
                normalized = (data - mean_val) / std_val
                normalized_data.append(normalized)
        else:
            raise ValueError("Normalization method must be 'min-max' or 'z-score'")
        return normalized_data

    @staticmethod
    def calculate_statistics(data_list):
        """
        Calculate statistics for each dataset in data_list.

        Returns:
            tuple: (means, quartile1, medians, quartile3, whiskers_min, whiskers_max)
        """
        means = []
        quartile1 = []
        medians = []
        quartile3 = []
        whiskers_min = []
        whiskers_max = []

        for data in data_list:
            sorted_data = np.sort(data)
            q1, med, q3 = np.percentile(sorted_data, [25, 50, 75])
            quartile1.append(q1)
            medians.append(med)
            quartile3.append(q3)
            whisker_min, whisker_max = PlotTest.adjacent_values(sorted_data, q1, q3)
            whiskers_min.append(whisker_min)
            whiskers_max.append(whisker_max)
            means.append(np.mean(data))

        return means, quartile1, medians, quartile3, whiskers_min, whiskers_max

    @staticmethod
    def plot_histograms(data_list, labels, means, title_prefix, image_path):
        """
        Plot histograms for each dataset.

        Parameters:
            data_list (list of arrays): Data to plot.
            labels (list of str): Labels for each dataset.
            means (list of float): Mean values for each dataset.
            title_prefix (str): Prefix for the plot title.
            image_path (str): Path to the image for the legend.
        """
        class HandlerImage(HandlerBase):
            def __init__(self, image, text, zoom=1):
                self.image = image
                self.text = text
                self.zoom = zoom
                HandlerBase.__init__(self)

            def create_artists(self, legend, orig_handle,
                               xdescent, ydescent, width, height, fontsize, trans):
                imagebox = OffsetImage(self.image, zoom=self.zoom)
                x_image = xdescent + width / 2
                y_image = ydescent + height  

                imagebox.set_offset((x_image, y_image))
                imagebox.set_transform(trans)

                text_x = xdescent + width / 2
                text_y = ydescent + height / 2  
                text_artist = plt.Text(
                    text_x, text_y, self.text, ha='center', va='top', fontsize=fontsize
                )
                text_artist.set_transform(trans)

                return [imagebox, text_artist]

        all_counts = []
        num_bins = 100
        maximum = 0

        for data in data_list:
            n, bins = np.histogram(data, bins=num_bins)
            max_count = n.max()
            maximum = max(max_count, maximum)

        for index, data in enumerate(data_list):
            plt.figure()
            n, bins, patches = plt.hist(data, bins=num_bins, alpha=0.5)
            median = np.median(data)

            image_proxy = Line2D([], [], linestyle='none')

            image = mpimg.imread(image_path)

         
            legend_handles = [image_proxy]
            legend_labels = [''] 

            handler_map = {
                image_proxy: HandlerImage(
                    image, f'{labels[index]} Mean: {means[index]:.4f}', 0.05
                )
            }

            plt.legend(
                handles=legend_handles,
                labels=legend_labels,
                handler_map=handler_map,
                loc='upper right'
            )

            if title_prefix != 'Distortion' and title_prefix != 'Angle':
                plt.xlim(-0.2, 1.2)
                plt.xlabel('Normalized Value')
            else:
                plt.xlabel('Value')

            plt.ylim(0, maximum * 1.1)
            plt.ylabel('Frequency')
            plt.title(f'{title_prefix} - {labels[index]}')
            plt.axvline(x=median, color='r', linestyle='dashed', linewidth=1)
            plt.savefig(
                f'{title_prefix.lower().replace(" ", "_")}_{labels[index]}.png'
            )
            plt.close()

    @staticmethod
    def plot_violin(data_list, labels, medians, quartile1, quartile3, whiskers_min, whiskers_max, title, is_normalized=True):
        """
        Plot violin plot for the data.

        Parameters:
            data_list (list of arrays): Data to plot.
            labels (list of str): Labels for each dataset.
            medians, quartile1, quartile3, whiskers_min, whiskers_max: Statistics for the data.
            title (str): Title for the plot.
            is_normalized (bool): Whether the data is normalized.
        """
        fig, ax = plt.subplots(figsize=(9, 4))

        parts = ax.violinplot(
            data_list, showmeans=False, showmedians=False, showextrema=False
        )

        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')  
            pc.set_edgecolor('black')   
            pc.set_alpha(1)

        inds = np.arange(1, len(medians) + 1)
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        PlotTest.set_axis_style(ax, labels)
        if is_normalized:
            ax.set_ylabel('Normalized Value')
        else:
            ax.set_ylabel('Value')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()

    def plot_metrics(self, results, metric_name):
        data_list = []
        labels = []

        for rez in results:
            print(f"Processing result: {rez[1]}")
            print(f"Type of rez[0]: {type(rez[0])}")
            print(f"Content of rez[0]: {rez[0]}")

            try:
                if isinstance(rez[0], (list, tuple)):
                    data = np.concatenate([np.array(sublist).flatten() for sublist in rez[0]])
                elif isinstance(rez[0], np.ndarray):
                    data = rez[0].flatten()
                else:
                    raise ValueError(f"Unsupported data type in rez[0]: {type(rez[0])}")

                if not np.issubdtype(data.dtype, np.number):
                    raise ValueError(f"Data in rez[0] must be numerical. Found {data.dtype}.")

                data_list.append(data)
                labels.append(rez[1])

            except Exception as e:
                print(f"Error processing {rez[1]}: {e}")
                continue  

        if not data_list:
            raise ValueError("No valid data to plot.")

        if metric_name != 'Distortion' and metric_name != 'Angle':
            normalized_data = self.normalize_data(data_list, method='min-max')
            is_normalized = True
        else:
            normalized_data = data_list
            is_normalized = False

        means, quartile1, medians, quartile3, whiskers_min, whiskers_max = self.calculate_statistics(normalized_data)

        self.plot_histograms(normalized_data, labels, means, metric_name, "Linear_RGB_color_wheel.png")

        self.plot_violin(
            normalized_data, labels, medians, quartile1, quartile3, whiskers_min, whiskers_max,
            f'violin plot {metric_name}', is_normalized
        )

if __name__ == "__main__":
    plotter = PlotTest()

    # Example data (you need to provide actual data)
    # results = [
    #     (np.random.normal(0, 1, 1000), 'Algorithm 1'),
    #     (np.random.normal(0, 1, 1000), 'Algorithm 2'),
    #     (np.random.normal(0, 1, 1000), 'Algorithm 3'),
    # ]

    # Plotting
    # plotter.plotDistortionHistogram(results)
