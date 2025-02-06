import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sbn
from matplotlib.lines import Line2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerBase
from plotnine import ggplot, aes, geom_violin, geom_boxplot, theme, element_text, labs, element_blank


class PlotTest:
    def __init__(self, output_folder="."):
        """
        :param output_folder: Base directory where final plots will be saved.
        """
        self.output_folder = output_folder

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

    def plotSelfIntersect(self, results):
        self.plot_metrics(results, 'Self Intersect')

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
        Normalize the data using the specified method: 'min-max' or 'z-score'.
        """
        normalized_data = []
        if method == 'min-max':
            all_data = np.concatenate(data_list)
            min_val = np.min(all_data)
            max_val = np.max(all_data)
            for data in data_list:
                if (max_val - min_val) != 0:
                    normalized = (data - min_val) / (max_val - min_val)
                else:
                    normalized = data  # fallback if all_data is constant
                normalized_data.append(normalized)
        elif method == 'z-score':
            all_data = np.concatenate(data_list)
            mean_val = np.mean(all_data)
            std_val = np.std(all_data)
            for data in data_list:
                if std_val != 0:
                    normalized = (data - mean_val) / std_val
                else:
                    normalized = data  # fallback if std=0
                normalized_data.append(normalized)
        else:
            raise ValueError("Normalization method must be 'min-max' or 'z-score'")
        return normalized_data

    @staticmethod
    def calculate_statistics(data_list):
        """
        Calculate statistics for each dataset in data_list.

        Returns:
          (means, quartile1, medians, quartile3, whiskers_min, whiskers_max)
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
            w_min, w_max = PlotTest.adjacent_values(sorted_data, q1, q3)
            whiskers_min.append(w_min)
            whiskers_max.append(w_max)
            means.append(np.mean(data))

        return means, quartile1, medians, quartile3, whiskers_min, whiskers_max

    def plot_histograms(self, data_list, labels, means, title_prefix, image_path,
                        x_is_normalized=False):
        """
        Plot histograms for each dataset, ensuring the same X- and Y-limits.

        :param data_list: list of arrays (one array per algorithm)
        :param labels: list of str (one label per algorithm)
        :param means: list of floats (mean value of each dataset)
        :param title_prefix: e.g. 'Distortion'
        :param image_path: path to an image used for the legend (optional usage)
        :param x_is_normalized: bool, if True we clamp X from [-0.2, 1.2]
        """
        # Optional: If you don't want an embedded image in the legend, remove HandlerImage usage below.
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

        num_bins = 50

        # 1) Determine global X-limits (and optionally clamp for normalized metrics).
        if x_is_normalized:
            # For normalized metrics, we'll just clamp to [-0.2, 1.2]
            global_x_min, global_x_max = -0.2, 1.2
        else:
            # Freed: base on actual data
            global_x_min = min(d.min() for d in data_list)
            global_x_max = max(d.max() for d in data_list)

        # 2) First pass: gather histograms to find global max count (Y-limit).
        hist_results = []
        global_max_count = 0
        for data in data_list:
            # np.histogram so we can see the counts (without plotting)
            n, bins = np.histogram(data, bins=num_bins, range=(global_x_min, global_x_max))
            global_max_count = max(global_max_count, n.max())
            hist_results.append((n, bins))

        # 3) Second pass: actually create each histogram with consistent axis limits.
        for index, (n, bins) in enumerate(hist_results):
            label = labels[index]
            data = data_list[index]

            # Create a subfolder for this label
            label_folder = os.path.join(self.output_folder, label)
            os.makedirs(label_folder, exist_ok=True)

            plt.figure()
            # Plot with the same bins so x-limits are consistent
            plt.hist(data, bins=bins, alpha=0.5, color='blue', range=(global_x_min, global_x_max))

            median = np.median(data)

            # If you want a custom legend with an image:
            if os.path.isfile(image_path):
                image = mpimg.imread(image_path)
                image_proxy = Line2D([], [], linestyle='none')

                legend_handles = [image_proxy]
                legend_labels = ['']
                handler_map = {
                    image_proxy: HandlerImage(
                        image, f'{label} Mean: {means[index]:.4f}', 0.05
                    )
                }

                plt.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    handler_map=handler_map,
                    loc='upper right'
                )

            # Set X-limits
            plt.xlim(global_x_min, global_x_max)
            if x_is_normalized:
                plt.xlabel('Normalized Value')
            else:
                plt.xlabel('Value')

            # Set the Y-limits to a bit more than global_max_count
            plt.ylim(0, global_max_count * 1.1)
            plt.ylabel('Frequency')
            plt.title(f'{title_prefix} - {label}')
            plt.axvline(x=median, color='r', linestyle='dashed', linewidth=1)

            # Build the output file path
            out_name = f"{title_prefix.lower().replace(' ', '_')}_{label}.png"
            out_path = os.path.join(label_folder, out_name)

            plt.savefig(out_path)
            plt.close()

    def plot_violin(self, data_list, labels, medians, quartile1,
                    quartile3, whiskers_min, whiskers_max, title,
                    is_normalized=True):
        """
        Plot a single violin plot containing all labels,
        with consistent labeling. The figure is saved at the top-level
        of self.output_folder.
        """
        fig, ax = plt.subplots(figsize=(9, 4))
        parts = ax.violinplot(
            data_list, showmeans=False, showmedians=False, showextrema=False
        )

        # Style the violin bodies
        for pc in parts['bodies']:
            pc.set_facecolor('#D43F3A')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        inds = np.arange(1, len(medians) + 1)
        # Plot medians as white dots
        ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
        # Plot quartile ranges
        ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
        # Optionally plot whiskers:
        # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        self.set_axis_style(ax, labels)
        ax.set_ylabel('Normalized Value' if is_normalized else 'Value')
        ax.set_title(title)

        plt.tight_layout()

        filename = f"{title.lower().replace(' ', '_')}.png"
        out_path = os.path.join(self.output_folder, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()

    def plot_metrics(self, results, metric_name):
        """
        Master function to handle a list of (values, label) pairs and produce
        per-label histograms + a single combined violin plot.
        """
        data_list = []
        labels = []

        for rez in results:
            print(f"Processing result: {rez[1]}")
            # rez[0] is the data array/list, rez[1] is the label
            try:
                # Flatten any sub-lists or arrays
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

        # If metric_name is not Distortion or Angle, we do min-max normalization by default
        is_normalized = (metric_name not in ['Distortion', 'Angle'])

        if is_normalized:
            normalized_data = self.normalize_data(data_list, method='min-max')
        else:
            normalized_data = data_list

        means, quartile1, medians, quartile3, whiskers_min, whiskers_max = self.calculate_statistics(normalized_data)

        # 1) Plot histograms for each label (algorithm) in its own subfolder
        color_wheel_img_path = "Linear_RGB_color_wheel.png"  # or any existing image

        self.plot_histograms(normalized_data, labels, means,
                             title_prefix=metric_name,
                             image_path=color_wheel_img_path,
                             x_is_normalized=is_normalized)

        # 2) Plot a single combined violin for all labels at once
        self.plot_violin(
            normalized_data,
            labels,
            medians,
            quartile1,
            quartile3,
            whiskers_min,
            whiskers_max,
            f'violin plot {metric_name}',
            is_normalized
        )


if __name__ == "__main__":
    # Example usage:
    plotter = PlotTest(output_folder="sample_output")

    # Suppose we have some fake data:
    results_example = [
        (np.random.normal(0, 1, 1000), 'fd'),
        (np.random.normal(5, 2, 1000), 'epb'),
        (np.random.normal(-2, 1, 1000), 'wr'),
    ]

    # Let's test with a "Distortion" metric (NO normalization).
    plotter.plotDistortionHistogram(results_example)

    # Test with "Monotonicity" (which triggers min-max normalization).
    plotter.plotMonotonicity(results_example)

    print("Plots saved in 'sample_output/'.\n"
          "Check subfolders for label-specific histograms, and top-level for violin plots.")
