from calvin_utils.statistical_utils.distribution_statistics import bootstrap_distribution_statistics
from math import pi

import os 
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from calvin_utils.plotting_utils.simple_heatmap import simple_heatmap
from sklearn.metrics import accuracy_score, auc, roc_curve, accuracy_score, confusion_matrix, precision_recall_fscore_support, precision_recall_curve, average_precision_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import roc_auc_score


class BinaryDataMetricsPlotter:
    def __init__(self, dataframe, mapping_dict, specified_metrics=None, out_dir=None, cm_normalization=None):
        """
        Initialize with a dataframe containing binary data and a dictionary mapping columns.
        """
        self.dataframe = dataframe
        self.mapping_dict = mapping_dict
        self.specified_metrics = specified_metrics
        self.save_dir = out_dir
        self.metrics = self.calculate_metrics()
        self.confusion_matrices = self.get_confusion_matrices(normalize=cm_normalization)
        
    def calculate_metrics(self):
        metrics = {}
        for truth, prediction in self.mapping_dict.items():
            tn, fp, fn, tp = confusion_matrix(self.dataframe[truth], self.dataframe[prediction]).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) != 0 else 0  # Positive Predictive Value
            npv = tn / (tn + fn) if (tn + fn) != 0 else 0  # Negative Predictive Value
            acc = accuracy_score(self.dataframe[truth], self.dataframe[prediction])
            precision, recall, f1, _ = precision_recall_fscore_support(self.dataframe[truth], self.dataframe[prediction], average='binary')

            metrics[(truth, prediction)] = {
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Precision': precision,
                'PPV': ppv,
                'NPV': npv,
                'Accuracy': acc,
                'F1 Score': f1,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn
            }
        return metrics
    
    def get_confusion_matrices(self, normalize=False):
        confusion_matrices = {}
        for ground_truth, predicted in self.mapping_dict.items():
            cm = confusion_matrix(self.dataframe[ground_truth], self.dataframe[predicted], normalize=normalize)
            confusion_matrices[(ground_truth, predicted)] = cm
        return confusion_matrices

    def plot_confusion_matrices(self):
        confusion_matrices = self.confusion_matrices
        num_matrices = len(confusion_matrices)
        fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 6))
        
        if num_matrices == 1:
            axes = [axes]
            
        for ax, ((ground_truth, predicted), cm) in zip(axes, confusion_matrices.items()):
            simple_heatmap(
                cm,
                dataset_name="",
                ax=ax,
                palette="Blues",
                annot=True,
                fmt=".2f",
                labels=['Negative', 'Positive'],
            )
            ax.set_ylabel(f'Predicted: {predicted}', fontsize=16)
            ax.set_xlabel(f'Actual: {ground_truth}', fontsize=16)
            ax.set_title(f'Confusion Matrix for {ground_truth} vs {predicted}', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            
        if self.save_dir is not None:
            subdir = "confusion_matrix"
            os.makedirs(os.path.join(self.save_dir, subdir), exist_ok=True)
            file_name_svg = "conf_matrix.svg"
            path_svg = os.path.join(self.save_dir, subdir, file_name_svg)
            plt.savefig(path_svg, format='svg')
            
        plt.tight_layout()
        plt.show()
        
    def plot_radar_charts(self):
        if self.specified_metrics is None:
            self.specified_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
            
        tab10 = sns.color_palette("tab10", 10)
        color_map = sns.color_palette([tab10[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
        
        for idx, ((old_col, new_col), metric_values) in enumerate(self.metrics.items()):
            plt.figure(figsize=(6, 6))
            ax = plt.subplot(111, polar=True)

            categories = self.specified_metrics
            N = len(categories)

            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            ax.set_theta_offset(pi / 2)
            ax.set_theta_direction(-1)

            plt.xticks(angles[:-1], categories)

            ax.set_rlabel_position(0)
            plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2","0.4","0.6","0.8"], color="black", size=12)
            plt.ylim(0,1)

            values = [metric_values[metric] for metric in self.specified_metrics]
            values += values[:1]
            ax.plot(angles, values, linewidth=1, linestyle='solid', label=f'{old_col} to {new_col}', color=color_map[idx])
            ax.fill(angles, values, alpha=0.25, color=color_map[idx])

            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title(f'Metrics for "{old_col} to {new_col}"', size=15, color='black', y=1.1)

            if self.save_dir is not None:
                radar_plots_subdir = "radar_plots"
                os.makedirs(os.path.join(self.save_dir, radar_plots_subdir), exist_ok=True)
                file_name_svg = f"{old_col}_to_{new_col}_radar.svg"
                path_svg = os.path.join(self.save_dir, radar_plots_subdir, file_name_svg)
                plt.savefig(path_svg, format='svg')
            plt.show()

            plt.close()

    def plot_metrics(self):
        if self.specified_metrics is None:
            self.specified_metrics = ['Sensitivity', 'Specificity', 'Precision', 'PPV', 'NPV', 'Accuracy', 'F1 Score']

        plot_data = []
        for (old_col, new_col), metric_values in self.metrics.items():
            for metric_name, metric_value in metric_values.items():
                if metric_name in self.specified_metrics:
                    plot_data.append({
                        'Mapping': f'{old_col} to {new_col}',
                        'Metric': metric_name,
                        'Value': metric_value
                    })

        plot_df = pd.DataFrame(plot_data)

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Value", y="Mapping", hue="Metric", data=plot_df)

        plt.xlabel('Metric Value')
        plt.ylabel('Column Mapping')
        plt.title('Performance Metrics for Each Column Mapping')

        plt.legend()
        plt.tight_layout()
        if self.save_dir is not None:
            subdir = "bar_plots"
            os.makedirs(os.path.join(self.save_dir, subdir), exist_ok=True)
            file_name_svg = f"{old_col}_to_{new_col}_bar.svg"
            path_svg = os.path.join(self.save_dir, subdir, file_name_svg)
            plt.savefig(path_svg, format='svg')

        plt.show()
    
    def convert_metrics_to_dataframe(self):
        """
        Converts the provided metrics dictionary into a pandas DataFrame.

        Args:
        metrics_dict (dict): A dictionary where each key is a tuple containing two strings
                            (categories) and each value is another dictionary containing
                            various metrics.

        Returns:
        pandas.DataFrame: A DataFrame with the metrics organized in columns and categories in rows.
        """
        import pandas as pd

        # Convert the dictionary to a DataFrame
        df = pd.DataFrame(self.metrics).T

        # Setting the names for the multi-index and resetting it to make it part of the DataFrame
        df.columns.name = 'Metric'
        df.index.set_names(['Category', 'Subcategory'], inplace=True)
        df.reset_index(inplace=True)
        
        if self.save_dir is not None:
            subdir = "metrics_df"
            os.makedirs(os.path.join(self.save_dir, subdir), exist_ok=True)
            df.to_csv(os.path.join(self.save_dir, subdir, 'metrics.csv'))

        return df
    
    def lineplot_metrics(self):
        # Convert metrics to DataFrame
        metrics_df = self.convert_metrics_to_dataframe()
        
        # Set up the color palette
        palette = sns.color_palette("tab10", 5)
        
        # Initialize the plot
        plt.figure(figsize=(6*len(metrics_df.index), 6))
        
        # Plot each metric
        sns.lineplot(x='Category', y='Accuracy', data=metrics_df, marker='o', label='Accuracy', color=palette[0])
        sns.lineplot(x='Category', y='Sensitivity', data=metrics_df, marker='o', label='Sensitivity', color=palette[1])
        sns.lineplot(x='Category', y='Specificity', data=metrics_df, marker='o', label='Specificity', color=palette[2])
        sns.lineplot(x='Category', y='PPV', data=metrics_df, marker='o', label='PPV', color=palette[3])
        sns.lineplot(x='Category', y='NPV', data=metrics_df, marker='o', label='NPV', color=palette[4])
        
        # Customize the plot
        plt.ylim(0, 1.05)
        plt.xlabel('Class', fontsize=20)
        plt.ylabel('Classification Metric Score', fontsize=20)
        plt.title('Classification Metrics Across Classes', fontsize=20)
        
        plt.xticks(fontsize=16)
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16)
        plt.legend(fontsize=16)
        
        plt.grid(False)
        sns.despine()
        
        if self.save_dir is not None:
            subdir = "metrics_lineplot"
            os.makedirs(os.path.join(self.save_dir, subdir), exist_ok=True)
            file_name_svg = "lineplot.svg"
            path_svg = os.path.join(self.save_dir, subdir, file_name_svg)
            plt.savefig(path_svg, format='svg')
        # Show the plot
        plt.tight_layout()
        plt.show()
            
    def plot_macro_averages(self):
        # Convert metrics to DataFrame
        metrics_df = self.convert_metrics_to_dataframe()
        
        # Calculate macro-averages and standard deviations
        metric_names = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']
        averages = metrics_df[metric_names].mean()
        std_devs = metrics_df[metric_names].std()
        print("Macro Averages: ", averages)
        print("Macro Standard Deviations: ", std_devs)

        # Create a DataFrame for plotting
        macro_df = pd.DataFrame({
            'Metric': metric_names,
            'Average': averages,
            'StdDev': std_devs
        })

        # Initialize the plot
        plt.figure(figsize=(18, 6))

        # Create bar plot with error bars
        sns.barplot(x='Metric', y='Average', yerr=macro_df['StdDev'], data=macro_df, palette='tab10', capsize=0.5)

        # Customize the plot
        plt.ylim(0, 1.05)
        plt.xlabel('Metric', fontsize=20)
        plt.ylabel('Macro-Average Score', fontsize=20)
        plt.title('Macro-Average Classification Metrics with Standard Deviations', fontsize=20)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.grid(False)
        sns.despine()
        
        if self.save_dir is not None:
            subdir = "macro_averages"
            os.makedirs(os.path.join(self.save_dir, subdir), exist_ok=True)
            file_name_svg = "macro_averages.svg"
            path_svg = os.path.join(self.save_dir, subdir, file_name_svg)
            plt.savefig(path_svg, format='svg')
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    def run_plotting(self):
        self.plot_metrics()
        self.plot_confusion_matrices()
        self.plot_radar_charts()
        self.lineplot_metrics()
        if len(self.mapping_dict.values()) > 1:
            self.plot_macro_averages()
        

class MulticlassClassificationEvaluation:
    """
    This is a class which will either take a fitted Statsmodels Model Object and a dataframe of observations,
    or it will take a dataframe of predictions and a dataframe of observations. 
    
    It will expect observations dataframeto take the form where actuals are one-hot encoded:
    [[0, 1, 0, 0],
      1, 0 ,0 ,0]]
      
    It will expect predictions dataframeto take the form where prediction for a given classificition is an array of probability:
    [[0.2, 0.7, 0.1, 0.0],
      0.9, 0.05, 0.05, 0.0]]
    
    Will extract various metrics such as accuracy, sensitivity, specificity, PPV, and NPV.
    Will generate a heatmap, normalized as you please.
    
    Notes:
    1. No Normalization:
        Description: The confusion matrix contains the raw counts of predictions for each combination of true and predicted classes.
        When to Use: When you're interested in the absolute numbers of observations for each prediction outcome.
    2. Normalization by True Classes ('true'):
        Description: Each element in the confusion matrix is divided by the sum of elements in its corresponding true class row. This results in each row summing to 1.
        Interpretation: The values represent the proportion of the actual (true) class instances that were predicted as each class. It's useful for understanding the distribution of predictions for each true class.
        When to Use: To analyze the classifier's ability to correctly identify each true class, especially when the distribution of classes is imbalanced.
    3. Normalization by Predicted Classes ('pred'):
        Description: Each element in the confusion matrix is divided by the sum of elements in its corresponding predicted class column. This results in each column summing to 1.
        Interpretation: The values indicate the proportion of predictions for each class that were actually instances of the true classes. It helps in assessing the precision or positive predictive value of predictions for each class.
        When to Use: To evaluate how confidently the model predicts each class, particularly when some classes are prone to being overpredicted.
    4. Normalization by All Elements ('all'):
        Description: Each element in the confusion matrix is divided by the total number of observations. This results in the entire matrix summing to 1.
        Interpretation: The values show the proportion of all observations that fall into each combination of true and predicted classes. It provides a holistic view of the classifier's performance across all classes.
        When to Use: When you want a high-level overview of the model's performance, including both the rate of correct predictions and the distribution of errors, relative to the entire dataset.
    
    Choosing a Normalization Method:
    The choice of normalization method depends on what 
    aspect of the model's performance you're most interested in:

    Use no normalization for a straightforward count of each outcome.
    Normalize by true classes ('true') to focus on sensitivity, recall, or the classifier's ability to identify each class.
    Normalize by predicted classes ('pred') to concentrate on precision or the likelihood that predicted instances of each class are correct.
    Normalize by all elements ('all') for a comprehensive view of the model's performance relative to the total number of observations.
    """
    
    def __init__(self, fitted_model, observation_df, normalization=None, predictions_df=None, out_dir=None, thresholds=None, assign_labels=True):
        """
        Initializes the ModelEvaluation with model results and the true outcomes.
        
        Args:
            fitted_model: The result object from a fitted statsmodels MNLogit model.
            observation_df: A pandas DataFrame with the true class outcomes in one-hot encoded format.
            normalization: Normalization method for the confusion matrix (None, 'true', 'pred', 'all').
            predictions_df: Manually entered DataFrame of predictions, can contain probabilities or dummy-coded predictions.
            thresholds (dict): a dictionary mapping the index of the threshold to the probability threshold to make that classification. 
            assign_labels (bool): Scipy's confusion matrix orders by minimum to maximum occurence of the predictions. It will output the confusion matrix by this. 
                                If set to False, we will organize our confusion matrix as per scipy's order. 
        """
        self.results = fitted_model
        self.outcome_matrix = observation_df
        self.normalization = normalization
        self.predictions_df = predictions_df
        self.out_dir = out_dir
        self.thresholds = thresholds
        self.assign_labels = assign_labels
    
    def relate_index_to_class(self):
        print("Note: The rasterized probability plot show the probability of the correect class by default.")
        for i, col in enumerate(self.outcome_matrix):
            print(f"{i}: {col}")
    
    def resolve_disagreements(self, probabilities, predictions, classes_above_threshold, relative_threshold=True, priority_rules=None, debug=False):
        """
        Resolves disagreements when multiple classes exceed their threshold.

        Args:
            probabilities (np.array): Probabilities for the current observation.
            predictions (np.array): Current predictions array to be updated.
            classes_above_threshold (np.array): Classes that have exceeded their threshold.
            relative_threshold (bool): Whether to consider relative thresholds for resolving disagreements.
            priority_rules (list of lists): Priority rules for resolving disagreements.
            debug (bool): If True, additional debug information will be printed.

        Returns:
            int: The index of the winning class after resolving disagreements.
        """
        winner_index = predictions  # Default to current predictions if no changes are made

        # Apply priority rules if any
        if priority_rules is not None:
            for rule in priority_rules:
                if set(rule).issubset(set(classes_above_threshold)):
                    winner_index = rule[-1]  # Apply priority rule
                    if debug:
                        print(f"Applied priority rule: {rule}, selected class {winner_index}")
                    return winner_index  # Return early if a priority rule is applied

        # If relative_threshold is True, and no priority rule was applied
        if relative_threshold:
            # Calculate relative scores as the probability of each class divided by its threshold
            relative_scores = probabilities[classes_above_threshold] / np.array([self.thresholds[cls] for cls in classes_above_threshold])
            winner_index = classes_above_threshold[np.argmax(relative_scores)]
            if debug:
                print(f"Relative scores considered. Winner: Class {winner_index} with scores: {relative_scores}")

        return winner_index
    
    def apply_manual_thresholds(self, debug=False):
        """
        Applies manual thresholds to the predictions based on argmax and predefined rules.
        Returns an array of the same shape as raw_predictions.argmax(1).
        
        The self.thresholds structure should map from class indices (as identified by argmax) to a function
        that adjusts the predicted class based on some condition.
        
        Example of how to format self.thresholds to generate rules: 
        Let's say we have 3 classes and we want to modify class_1 based on probabilities
        The index identified by argmax corresponds to the first key (0,1,2), which correspond to the class. 
        thresholds = {
            0: lambda probs: 0 if probs[0] > 0.5 else (1 if probs[0] > 0.25 else 2),  # Adjust class_0 predictions
            1: lambda probs: None,  # No threshold adjustment for class_1
            2: lambda probs: None   # No threshold adjustment for class_2
        }
        """        
        if len(self.raw_predictions.shape)==1:                          # binomial case
            classifications = (self.raw_predictions > 0.5).astype(int).flatten()
        else:                                                           # multinomial case
            classifications = self.raw_predictions.argmax(1)
        final_predictions = np.zeros(classifications.shape[0], dtype=int)

        for i, choice in enumerate(classifications):
            if choice in self.thresholds.keys():
                function = self.thresholds[choice]
                probabilities = self.raw_predictions[i]
                new_choice = function(probabilities)
                final_predictions[i] = new_choice
                if debug: print(f"Choice: {choice}, Probabilities: {probabilities}, New Choice: {new_choice}")
        return final_predictions 
    
    def get_predictions(self, debug=False):
        """
        Takes a model or a DF of probabilities (or dummy-coded predictions) and returns the prediction. 
        """
        if self.predictions_df is not None:
            self.raw_predictions = self.predictions_df.to_numpy()
        else:
            self.raw_predictions = self.results.predict()
            
        if self.thresholds is None:
            print("Taking maximum probability as prediction.")
            if len(self.raw_predictions.shape) == 1:                     # Account for binomial vs multinomial prediction structure
                self.predictions = (self.raw_predictions > 0.5).astype(int).flatten() # Binarize by 0.50, the default logistic decision curve threshold
            else:                                                           # Run the multinomial approach
                self.predictions = self.raw_predictions.argmax(1)           # Take max estimate
        else:
            print("Applying prescribed thresholds for prediction.")
            self.predictions = self.apply_manual_thresholds()
        self.predictions_df = pd.DataFrame(self.raw_predictions, columns=self.outcome_matrix.columns)
    
    def get_observations(self):
        """
        Takes a DF of dummy-coded observations and 
        """
        self.raw_observations = self.outcome_matrix.to_numpy()
        if len(self.raw_predictions.shape) == 1:                             # Binomial Observations Case
            self.observations = self.raw_observations.astype(int).flatten().tolist() 
        else:                                                               # Multinomial Observations Case
            self.observations = self.raw_observations.argmax(1) 
        self.observations_df = pd.DataFrame(self.raw_observations, columns = self.outcome_matrix.columns)
        for col in self.observations_df.columns:
            print(f"There are {np.sum(self.observations_df[col])} observations for {col}")
        
    def rasterized_probability_plot(self, probability_of_correct_class=True):
        """
        Plots rasterized probability plots for correct and incorrect classifications.
        The top subplot shows correct classifications, and the bottom subplot shows incorrect classifications.
        Each row in a subplot corresponds to a class. Incorrect classifications are color-coded by the correct class
        and placed in the row of the predicted class.
        If probability_of_correct_class is True, plot the probability of the true class for incorrect predictions.
            If false, will plot the probability of the selected class. 
        """
        n_classes = self.outcome_matrix.shape[1]
        colors = sns.color_palette("tab10") if n_classes <= 10 else sns.color_palette("tab20", n_classes)
        
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Loop through each prediction
        for i, (true, pred, probs) in enumerate(zip(self.observations, self.predictions, self.raw_predictions)):
            # Plotting the correct classifications
            if true == pred:    
                if len(self.raw_predictions.shape) == 1:                                                              # Binomial Case
                    axs[0].eventplot([probs], lineoffsets=pred, linelengths=0.5, colors=[colors[pred]])
                else:                                                                                           # Multinomial Case
                    axs[0].eventplot([probs[pred]], lineoffsets=pred, linelengths=0.5, colors=[colors[pred]])
            
            # Plotting the incorrect classifications
            else:
                # Choose the probability to plot: predicted class's probability or true class's probability
                if len(self.raw_predictions.shape) == 1:                                                              # Binomial Case
                    prob_to_plot = probs
                else:                                                                                           # Multinomial Case
                    prob_to_plot = probs[true] if probability_of_correct_class else probs[pred]
                axs[1].eventplot([prob_to_plot], lineoffsets=pred, linelengths=0.5, colors=[colors[true]])

        axs[0].set_title('Correct Classifications')
        axs[1].set_title('Incorrect Classifications')
        axs[1].set_xlabel('Probability')

        # Set the y-ticks for both subplots
        for ax in axs:
            ax.set_yticks(range(n_classes))
            if self.assign_labels:
                labels = list(self.outcome_matrix.columns)
                # Conditionally handle the binomial case
                if n_classes == 1:  # Binomial case, only one class label
                    positive_label = labels[0]
                    negative_label = f"not {positive_label}"
                    labels = [negative_label, positive_label]  # Create both labels for binomial case
                    ax.set_yticks([0, 1])  # Ensure ticks match the two labels
                ax.set_yticklabels(labels)
            else:
                ax.set_yticklabels([f'Class {i}' for i in range(n_classes)])

        # Create a legend for the colors
        custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in range(n_classes)]
        if self.assign_labels:
            axs[1].legend(custom_lines, [f'{self.outcome_matrix.columns[i]}' for i in range(n_classes)], title='True Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axs[1].legend(custom_lines, [f'Class {i}' for i in range(n_classes)], title='True Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        if self.out_dir:
            plt.savefig(os.path.join(self.out_dir, "rasterized_probabilities.png"))
            plt.savefig(os.path.join(self.out_dir, "rasterized_probabilities.svg"))
        plt.show()
        
    def handle_nans(self):
        self.predictions = np.nan_to_num(self.predictions)
        self.observations = np.nan_to_num(self.observations)
        self.predictions_df = self.predictions_df.fillna(value=0)
        self.observations_df = self.observations_df.fillna(value=0)
        self.raw_observations = np.nan_to_num(self.raw_observations)
        self.raw_predictions = np.nan_to_num(self.raw_predictions)
        self.outcome_matrix = self.outcome_matrix.fillna(value=0)
    
    def run(self):
        """Orchestrates the calculation and display of all evaluation metrics."""
        self.get_predictions()
        self.get_observations()
        self.rasterized_probability_plot()
        self.handle_nans()

class MulticlassOneVsAllROC(MulticlassClassificationEvaluation):
    """
    Extends MulticlassClassificationEvaluation to include ROC curve generation for each class
    in a multinomial logistic regression model.
    
    See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html fore more information.
    """
    def plot_radar_chart(self, metrics_df, classification='Micro-average', columns=['Acc', 'Sens', 'Spec', 'PPV', 'NPV']):
        """
        Generates and displays a radar chart for the calculated evaluation metrics.
        
        Parameters:
        metrics_df (pd.DataFrame): DataFrame containing the metrics for each class and the micro-average.
        out_dir (str, optional): Directory where the radar chart image will be saved. If None, the plot will not be saved.
        """
        # Select the micro-average row
        micro_avg_metrics = metrics_df[metrics_df['Class'] == classification].iloc[0]
        
        # Define the metrics and their corresponding angles on the radar chart
        values = [micro_avg_metrics[metric] for metric in columns]
        N = len(columns)

        # Repeat the first value to close the circle in the radar chart
        values += values[:1]

        # Calculate angle for each category
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the loop

        # Plot setup
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], columns, color='black', size=12)
        # Draw one axe per variable + add labels
        plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="black", size=10)
        plt.ylim(0, 1)
        # Plot data
        ax.plot(angles, values, linewidth=1, linestyle='solid', label='Micro-average Metrics')
        # Fill area
        ax.fill(angles, values, 'b', alpha=0.1)
        # Add a title and a legend and display the plot
        plt.title('Micro-average Model Evaluation Metrics', size=15, color='black', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        # Save plot if an output directory is specified
        if self.out_dir:
            plt.savefig(os.path.join(self.out_dir, "model_performance_radar_chart.svg"))
        plt.show()
        
    def plot_confusion_matrix(self):
        labels = self.observations_df.columns
        if len(labels) == 1:                                                        # Binomial case (one positive class)
            positive_label = labels[0]                                              # Assuming the first (and only) label is the positive case
            negative_label = f"not {positive_label}"
            labels = [negative_label, positive_label]
        conf_matrix = confusion_matrix(self.observations, self.predictions, normalize=self.normalization) #set the indices to actually be labels
        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        if self.normalization is None:
            digit_fmt = '1g'
        else:
            digit_fmt = '.2f'
        simple_heatmap(
            conf_matrix,
            dataset_name="",
            palette="viridis",
            annot=True,
            fmt=digit_fmt,
            labels=labels,
        )
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        if self.out_dir:
            plt.savefig(os.path.join(self.out_dir, 'confusion_matrix.svg'))
        plt.show()
        
        self.conf_matrix=conf_matrix
        
    def calculate_metrics(self):
        """
        Calculate performance metrics for each class and the micro-average from a confusion matrix.

        Returns:
        pd.DataFrame: DataFrame containing the metrics for each class and the micro-average.
        """
        metrics = {
            'Class': [],
            'TP': [],
            'FP': [],
            'TN': [],
            'FN': [],
            'Acc': [],
            'Sens': [],
            'Spec': [],
            'NPV': [],
            'PPV': [],
            'F1': []
            }
        # Get calculations for each class
        for i, col in enumerate(self.observations_df.columns):
            # Class specific confusion matrix
            TP = self.conf_matrix[i, i]
            FN = self.conf_matrix[i, :].sum() - TP
            FP = self.conf_matrix[:, i].sum() - TP
            TN = self.conf_matrix.sum() - (TP + FP + FN)
            
            # Class specific metrics
            Acc = (TP + TN) / (TP + FP + TN + FN)
            Sens = TP / (TP + FN) if (TP + FN) != 0 else 0
            Spec = TN / (TN + FP) if (TN + FP) != 0 else 0
            PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
            NPV = TN / (TN + FN) if (TN + FN) != 0 else 0
            F1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            
            metrics['Class'].append(col)
            metrics['TP'].append(TP)
            metrics['FP'].append(FP)
            metrics['TN'].append(TN)
            metrics['FN'].append(FN)
            metrics['Acc'].append(Acc)
            metrics['Sens'].append(Sens)
            metrics['Spec'].append(Spec)
            metrics['NPV'].append(NPV)
            metrics['PPV'].append(PPV)
            metrics['F1'].append(F1)
        
        # Calculate micro-average confusion matrix
        TP_micro = np.sum(metrics['TP'])
        FN_micro = np.sum(metrics['FN'])
        FP_micro = np.sum(metrics['FP'])
        TN_micro = np.sum(metrics['TN'])
        
        # Calculate micro-average classification metrics
        Acc_micro = (TP_micro + TN_micro) / (TP_micro + FP_micro + TN_micro + FN_micro)
        Sens_micro = TP_micro / (TP_micro + FN_micro) if (TP_micro + FN_micro) != 0 else 0
        Spec_micro = TN_micro / (TN_micro + FP_micro) if (TN_micro + FP_micro) != 0 else 0
        PPV_micro = TP_micro / (TP_micro + FP_micro) if (TP_micro + FP_micro) != 0 else 0
        NPV_micro = TN_micro / (TN_micro + FN_micro) if (TN_micro + FN_micro) != 0 else 0
        F1_micro = 2 * TP_micro / (2 * TP_micro + FP_micro + FN_micro) if (2 * TP_micro + FP_micro + FN_micro) != 0 else 0
        
        metrics['Class'].append('Micro-average')
        metrics['TP'].append(TP_micro)
        metrics['FP'].append(FP_micro)
        metrics['TN'].append(TN_micro)
        metrics['FN'].append(FN_micro)
        metrics['Acc'].append(Acc_micro)
        metrics['Sens'].append(Sens_micro)
        metrics['Spec'].append(Spec_micro)
        metrics['NPV'].append(NPV_micro)
        metrics['PPV'].append(PPV_micro)
        metrics['F1'].append(F1_micro)

        metrics_df = pd.DataFrame(metrics)
        return metrics_df

    def plot_roc_curves(self, silent=False):
        """
        Plots ROC curves for each class using a One-vs-Rest approach.
        For each class (i), observation_i and prediction_i are calculated.
        This describes how one class' predictions discriminate it from all other classes. 
        However, it neglects the fact that there may have been another prediction which was superior to it. 
        Thus, to address this, you need to do a micro-average and macro-average ROC, as this is not the full story. 
        """
        n_classes = self.outcome_matrix.shape[1]
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Binarize the output
        for i in range(n_classes):

            if len(self.raw_predictions.shape)==1:                                                      # binomial case
                fpr[i], tpr[i], _ = roc_curve(self.raw_observations, self.raw_predictions)
            else:                                                                                       # multinomial case
                fpr[i], tpr[i], _ = roc_curve(self.raw_observations[:, i], self.raw_predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot all ROC curves
        if not silent: 
            plt.figure()
            palette = sns.color_palette("tab10") if n_classes <= 10 else sns.color_palette("tab20", n_classes)
            for i, color in zip(range(n_classes), palette):
                try:
                    category = self.outcome_matrix.columns[i]
                except:
                    category = i
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'ROC curve of class {category}' + ' (area = {1:0.2f})'
                        ''.format(i, roc_auc[i]))
                
        # PLOT MICRO-AVERAGE ROC CURVE
        if self.results is not None:
            fpr, tpr, _ = roc_curve(self.outcome_matrix.to_numpy().ravel(), self.raw_predictions.ravel()) #??self.results.predict().ravel(). this was previously results.to_numpy().ravel().
        else:      
            # outcomes, predictions = self.prepare_micro_average_dfs()    
            fpr, tpr, thresholds = roc_curve(self.observations_df.to_numpy().ravel(), self.predictions_df.to_numpy().ravel())
        micro_roc_auc = auc(fpr, tpr)
        
        if not silent:
            plt.plot(fpr, tpr, label=f'Micro Avg. (area = {micro_roc_auc:0.2f})'
                    ''.format(roc_auc), color='k', linestyle=':', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('One-vs-Rest ROC Curves')
            plt.legend(loc="lower right")
            # Save plot if an output directory is specified
            if self.out_dir:
                os.makedirs(self.out_dir, exist_ok=True)
                plt.savefig(os.path.join(self.out_dir, "roc_ovr.png"))
                plt.savefig(os.path.join(self.out_dir, "roc_ovr.svg"))
            plt.show()
        return micro_roc_auc
        
    def find_optimal_thresholds(self):
        """
        Finds the optimal probability thresholds for each class using Youden's J statistic.
        """
        self.optimal_thresholds = {}
        for i in range(self.outcome_matrix.shape[1]):
            # Binarize the output for the current class 
            if len(self.raw_predictions.shape) == 1:                # Binomial Case
                true_bin = self.raw_observations
                prob_pred = self.raw_predictions
            else:                                                   # Multinomial case
                true_bin = self.raw_observations[:, i]
                prob_pred = self.raw_predictions[:, i]
            
            # Calculate the ROC curve
            fpr, tpr, thresholds = roc_curve(true_bin, prob_pred)
            
            # Calculate Youden's J statistic
            j_scores = tpr - fpr
            
            # Find the optimal threshold that maximizes Youden's J statistic
            optimal_index = np.argmax(j_scores)
            self.optimal_thresholds[i] = thresholds[optimal_index]
        print(f"Optimal Thresholds: \n {self.optimal_thresholds}")            
            
    def run(self):
        """
        Orchestrate the evaluation including confusion matrix, metrics, and ROC curves.
        """
        super().run()
        self.plot_roc_curves()
        self.find_optimal_thresholds()
        self.plot_confusion_matrix()
        metrics_df = self.calculate_metrics()
        self.plot_radar_chart(metrics_df)
        return metrics_df

class MacroAverageROC(MulticlassOneVsAllROC):
    """
    Extends MulticlassModelEvaluation to include the generation of a macro-average ROC curve
    for a multinomial logistic regression model.
    """
    def plot_macro_average_roc_curve(self):
        """
        Plots a macro-average ROC curve for the multinomial logistic regression model.
        
        This is like a meta-analytic ROC, but it averages all ROCs together with equal weight.
        This takes the AUROC for each class, then averages them. 
        """
        n_classes = self.outcome_matrix.shape[1]
        # Compute ROC curve and ROC area for each class
        all_fpr = np.unique(np.concatenate([np.linspace(0, 1, 100) for _ in range(n_classes)]))
        
        # Then interpolate all ROC curves at these points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            if len(self.raw_predictions.shape) == 1:                    # Binomial Case
                fpr, tpr, _ = roc_curve(self.outcome_matrix, self.raw_predictions) if self.results is not None else roc_curve(self.outcome_matrix, self.predictions_df)
            else:                                                       # Multinomial Case
                fpr, tpr, _ = roc_curve(self.outcome_matrix.iloc[:, i], self.raw_predictions[:, i]) if self.results is not None else roc_curve(self.outcome_matrix.iloc[:, i], self.predictions_df.iloc[:, i])
            
            mean_tpr += np.interp(all_fpr, fpr, tpr)
        
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        fpr = all_fpr
        tpr = mean_tpr
        roc_auc = auc(fpr, tpr)
        
        # Plot the macro-average ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='Macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc), color='navy', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Macro-average ROC Curve')
        plt.legend(loc="lower right")
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            plt.savefig(os.path.join(self.out_dir, "roc_macro.png"))
            plt.savefig(os.path.join(self.out_dir, "roc_macro.svg"))
        plt.show()

    def run(self):
        """
        Orchestrates the evaluation including the macro-average ROC curve.
        """
        super().run()
        if len(self.raw_predictions.shape) != 1: self.plot_macro_average_roc_curve()                 # only plot in Multinomial Case
        
class MicroAverageROC(MacroAverageROC):
    """
    Extends MacroAverageROC to include the generation of both macro-average and micro-average ROC curves
    for a multinomial logistic regression model.
    
    This is like a meta-analytic ROC. It averages the ROC curves, but with respect to sample weight. 
    See https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html fore more information.
    """
    def plot_micro_average_roc_curve(self):
        """
        Plots a micro-average ROC curve for the multinomial logistic regression model.
        """
        n_classes = self.outcome_matrix.shape[1]
        # Aggregate all false positive rates and true positive rates
        if self.results is not None:
            fpr, tpr, _ = roc_curve(self.outcome_matrix.to_numpy().ravel(), self.raw_predictions.ravel()) 
        else:      
            # outcomes, predictions = self.prepare_micro_average_dfs()    
            fpr, tpr, thresholds = roc_curve(self.observations_df.to_numpy().ravel(), self.predictions_df.to_numpy().ravel())
        
        roc_auc = auc(fpr, tpr)
        # Plot the micro-average ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label='Micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc), color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Micro-average ROC Curve')
        plt.legend(loc="lower right")
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            plt.savefig(os.path.join(self.out_dir, "roc_micro.png"))
            plt.savefig(os.path.join(self.out_dir, "roc_micro.svg"))
        plt.show()

    def run(self):
        """
        Orchestrates the evaluation including both the macro-average and micro-average ROC curves.
        """
        super().run()
        if len(self.raw_predictions.shape) != 1: self.plot_micro_average_roc_curve()                 # only plot in Multinomial Case
        
class MulticlassAUPRC(MicroAverageROC):
    '''
    for more info, see: https://glassboxmedicine.com/2019/03/02/measuring-performance-auprc/
    '''
    def plot_one_vs_all_auprc(self):
        """
        Plots the one-versus-all Precision-Recall curve for each class and includes iso-F1 curves.
        """
        n_classes = self.outcome_matrix.shape[1]
        colors = plt.get_cmap('tab10') if n_classes <= 10 else plt.get_cmap('tab20')
        
        plt.figure(figsize=(10, 8))
        
        # Compute PR curve and AUPRC for each class
        for i in range(n_classes):
            if self.results is not None:
                precision, recall, _ = precision_recall_curve(self.outcome_matrix.iloc[:, i], self.raw_predictions[:, i])
            else:
                precision, recall, _ = precision_recall_curve(self.outcome_matrix.iloc[:, i], self.predictions_df.iloc[:, i])
            expected = np.sum(self.outcome_matrix.iloc[:, i]) / self.outcome_matrix.shape[0]
            plt.plot(recall, precision, color=colors(i), lw=2, label=f'{self.outcome_matrix.columns[i]} (area = {auc(recall, precision):.2f} | expected = {expected:.2f})')
        
        # calculate the micro-average
        if self.results is not None:
            precision, recall, _ = precision_recall_curve(self.observations_df.to_numpy().ravel(),self.raw_predictions.ravel()) 
        else:      
            # outcomes, predictions = self.prepare_micro_average_dfs()    
            precision, recall, _ = precision_recall_curve(self.observations_df.to_numpy().ravel(), self.predictions_df.to_numpy().ravel())
        expected = np.sum(self.observations_df.to_numpy().ravel()) / len(self.observations_df.to_numpy().ravel())
        
        plt.plot(recall, precision,  color='k', lw=2, label=f'Micro Avg. (area = {auc(recall, precision):.2f} | expected = {expected:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim((0,1))
        plt.xlim((0,1))
        plt.title('One-Versus-All Precision-Recall Curves')
        plt.legend(loc='best')
        if self.out_dir:
            plt.savefig(os.path.join(self.out_dir, 'ova_auprc.svg'))
        plt.show()
        
        
    def plot_macro_average_auprc(self):
        """
        Plots a macro-average Precision-Recall curve for the multinomial logistic regression model.

        This averages all PR curves together with equal weight.
        """

        n_classes = self.outcome_matrix.shape[1]
        all_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros_like(all_recall)
        mean_expected = 0

        # Compute PR curve and AUPRC for each class
        for i in range(n_classes):
            
            if len(self.raw_predictions.shape)==1:                   # Binomial Case
                precision, recall, _ = precision_recall_curve(self.outcome_matrix, self.raw_predictions) if self.results is not None else precision_recall_curve(self.outcome_matrix, self.predictions_df)
                expected = np.sum(self.outcome_matrix) / self.outcome_matrix.shape[0]
            else:                                                    # Multinomial Case
                precision, recall, _ = precision_recall_curve(self.outcome_matrix.iloc[:, i], self.raw_predictions[:, i]) if self.results is not None else precision_recall_curve(self.outcome_matrix.iloc[:, i], self.predictions_df.iloc[:, i])
                expected = np.sum(self.outcome_matrix.iloc[:, i]) / self.outcome_matrix.shape[0]
            
            # Interpolate all PR curves at these points
            mean_precision += np.interp(all_recall, recall[::-1], precision[::-1])
            mean_expected += expected

        # Average it
        mean_expected /= n_classes
        mean_precision /= n_classes
        mean_auprc = auc(all_recall, mean_precision)

        # Calculate the number of true positives and observations
        tp = np.sum(self.outcome_matrix, axis=0)
        obs = self.outcome_matrix.shape[0]

        # Calculate F1 score based on TP/Obs
        f1_scores = tp / obs

        # Plot the macro-average PR curve
        plt.figure(figsize=(10, 8))
        if len(self.raw_predictions.shape) == 1: 
            plt.plot(all_recall, mean_precision, color='k', lw=2, label=f'Macro Avg. (area = {mean_auprc} | expected = {mean_expected})')
        else:
            plt.plot(all_recall, mean_precision, color='k', lw=2, label=f'Macro Avg. (area = {mean_auprc:.2f} | expected = {mean_expected:.2f})')

        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.ylim((0,1))
        plt.xlim((0,1))
        plt.title('Macro-Average Precision-Recall Curve')
        plt.legend(loc='best')
        if self.out_dir:
            plt.savefig(os.path.join(self.out_dir, 'macro_average_auprc.svg'))
        plt.show()
    
    def run(self):
        super().run()
        self.plot_macro_average_auprc()
        if len(self.raw_predictions.shape) != 1: self.plot_one_vs_all_auprc()           # only plot in Multinomial Case


class ComprehensiveMulticlassROC(MulticlassAUPRC):
    """
    **SEE THE DOCS FOR MulticlassClassificationEvaluation TO UNDERSTAND HOW TO CALL THIS**
    
    Extends ClassificationEvaluation to include ROC curve generation for a multinomial logistic regression model.
    
    ROC Curves Included:
    - First Curve: Individual ROC curves for the classification of each class against all other classes. This is 
      known as the One-vs-Rest (OvR) approach, where each class is considered as the positive class once, and 
      all other classes are combined to form the negative class, producing a ROC curve for each class.
      
    - Second Curve (Macro Average): A single ROC curve that represents the average performance across all classes. 
      The True Positive Rate (TPR) and False Positive Rate (FPR) are averaged for each class, treating each class 
      equally, regardless of its size or frequency in the dataset. This curve provides an overall performance measure 
      that gives equal weight to each class.
      
    - Third Curve (Micro Average): A single ROC curve that aggregates the contributions of all classes to compute the 
      overall TPR and FPR. This approach gives equal weight to each instance, summing the individual true positives, 
      false positives, true negatives, and false negatives across all classes before calculating TPR and FPR. The 
      micro-average ROC curve is especially useful in datasets with class imbalance, as it reflects the model's 
      performance across all instances, but biases towards more highly represented classes.
      
    """
    def save_dataframes(self):
        self.predictions_df.to_csv(self.out_dir+'/predicted_probabilities.csv', index=False)
        self.observations_df.to_csv(self.out_dir+'/observed_labels.csv', index=False)
        print("Saved predicted probabilities to CSV files at: ", self.out_dir, 'as predicted_probabilities.csv')
        print("and")
        print("Saved observed labels to CSV files at: ", self.out_dir, 'as observed_labels.csv')
        
    def run(self):
        """
        Orchestrates the evaluation including both the macro-average and micro-average ROC curves.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        super().run()

    def get_micro_auc(self):
        self.get_predictions()
        self.get_observations()
        micro_auc = self.plot_roc_curves(silent=True)
        return micro_auc
    
    @staticmethod
    def bootstrap_ovr_auroc(
        raw_observations: np.ndarray,
        raw_predictions: np.ndarray,
        outcome_matrix_cols,
        n_bootstraps: int = 1000,
        random_state: int = None,
        ci_alpha: float = 0.95
    ):
        """
        Bootstraps the one-vs-all AUROC for each class.

        Parameters
        ----------
        raw_observations : np.ndarray
            Ground truth one-hot encoded array of shape (n_samples, n_classes).
        raw_predictions : np.ndarray
            Predicted probabilities for each class of shape (n_samples, n_classes).
        outcome_matrix_cols : list-like
            The column (class) labels, typically from `self.outcome_matrix.columns`.
        n_bootstraps : int
            Number of bootstrap iterations.
        random_state : int
            Controls reproducibility of the random sampling.
        ci_alpha : float
            Confidence interval coverage (0 < ci_alpha < 1). 0.95 -> 95% CI.

        Returns
        -------
        auroc_summary_df : pd.DataFrame
            A table containing mean AUROC, standard deviation, and confidence
            intervals for each class.
        bootstrap_results : dict
            A dictionary with class labels as keys and lists of bootstrapped
            AUROC values as values.
        """
        rng = np.random.default_rng(seed=random_state)
        n_classes = raw_observations.shape[1]

        bootstrap_results = {}
        for i in range(n_classes):
            class_name = outcome_matrix_cols[i]
            y_true = raw_observations[:, i]
            y_score = raw_predictions[:, i]

            # Collect bootstrap results for this class
            boot_aurocs = []
            n_samples = len(y_true)
            for _ in range(n_bootstraps):
                idx = rng.integers(0, n_samples, n_samples)
                y_true_boot = y_true[idx]
                y_score_boot = y_score[idx]

                # If bootstrap sample has only one label, store NaN instead
                if np.unique(y_true_boot).size < 2:
                    boot_aurocs.append(np.nan)
                else:
                    boot_aurocs.append(roc_auc_score(y_true_boot, y_score_boot))
            print('Done ', class_name)
            bootstrap_results[class_name] = boot_aurocs

        # Build summary (mean, std, confidence intervals)
        alpha_lower = (1.0 - ci_alpha) / 2.0
        alpha_upper = 1.0 - alpha_lower
        rows = []
        for class_name, dist in bootstrap_results.items():
            dist_clean = [x for x in dist if not np.isnan(x)]
            if len(dist_clean) == 0:
                rows.append({
                    "Class": class_name,
                    "Mean AUROC": np.nan,
                    "Std": np.nan,
                    "lower_ci": np.nan,
                    "upper_ci": np.nan
                })
            else:
                dist_arr = np.array(dist_clean)
                mean_auroc = np.mean(dist_arr)
                std_auroc = np.std(dist_arr)
                lower_ci = np.quantile(dist_arr, alpha_lower)
                upper_ci = np.quantile(dist_arr, alpha_upper)
                rows.append({
                    "Class": class_name,
                    "Mean AUROC": mean_auroc,
                    "Std": std_auroc,
                    "lower_ci": lower_ci,
                    "upper_ci": upper_ci
                    })

            auroc_summary_df = pd.DataFrame(rows)
        return auroc_summary_df, bootstrap_results

    @staticmethod
    def plot_ovr_auc_with_ci(auroc_summary_df, x_label="AUC", y_label="Classes", ci_alpha=0.95, out_dir=None):
        """
        Plots the AUC scores for each class with 95% confidence intervals.

        Args:
            auroc_summary_df (pd.DataFrame): The summary DataFrame containing 'Class', 'Mean AUROC',
                                            'Std', and CI columns (e.g., "2.5%", "97.5%").
            x_label (str): Label for the x-axis (default: "AUC").
            y_label (str): Label for the y-axis (default: "Classes").
            ci_alpha (float): Confidence interval level (default: 0.95 for 95% CI).
        """
        # Ensure necessary columns exist in the DataFrame
        required_cols = ["Class", "Mean AUROC", "lower_ci", "upper_ci"]
        for col in required_cols:
            if col not in auroc_summary_df:
                raise ValueError(f"Missing column in DataFrame: {col}")
        
        # Sorting data by mean AUROC for better visualization
        sorted_df = auroc_summary_df.sort_values(by="Mean AUROC", ascending=False)
        
        # Initialize the plot
        plt.figure(figsize=(8, 6))
        sns.set(style="white")
        palette = sns.color_palette('tab10', n_colors=len(sorted_df))
        
        # Iterate through each row and plot
        for i, (row, color) in enumerate(zip(sorted_df.itertuples(), palette)):
            class_name = row.Class
            mean_auc = row._2  # Assuming "Mean AUROC" is the second column
            lower_ci = row.lower_ci  # Assuming lower CI is the third column
            upper_ci = row.upper_ci  # Assuming upper CI is the fourth column

            # Plot the confidence interval line
            plt.hlines(y=i, xmin=lower_ci, xmax=upper_ci, color=color, alpha=0.7)
            
            # Plot the mean AUC as a central dot
            plt.scatter(mean_auc, i, color=color, zorder=3)
            
        # Add a dashed grey line at AUC = 0.50
        plt.axvline(0.50, color='grey', linestyle='--', linewidth=1, alpha=0.8)
        
        # Set axis labels and ticks
        plt.yticks(range(len(sorted_df)), sorted_df["Class"])
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title("One-vs-All AUC with 95% Confidence Intervals", fontsize=14)
        
        plt.tight_layout()
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, 'bootstrapped_ovr_aurocs.svg'))
        plt.show()
        
    @staticmethod
    def plot_metric_with_ci(dfs, metric, out_dir):
        """
        Plots the specified metric with 95% confidence intervals for each disease.

        Args:
            dfs (dict): Dictionary containing DataFrames with 'Mean', '0.025%', and '0.975%' values.
            metric (str): The metric to plot (e.g., "Sensitivity", "Specificity", "NPV", "PPV", "Accuracy").
            out_dir (str): Directory where the plot should be saved.
        """
        plt.figure(figsize=(8, 6))
        palette = sns.color_palette('tab10', n_colors=len(dfs.items()))

        metric_means = []
        metric_lows = []
        metric_highs = []
        disease_labels = []

        for disease, df in dfs.items():
            mean_val = df.loc[metric, "Mean"]
            ci_low = df.loc[metric, "0.025%"]
            ci_high = df.loc[metric, "0.975%"]

            metric_means.append(mean_val)
            metric_lows.append(ci_low)
            metric_highs.append(ci_high)
            disease_labels.append(disease)

        # Create horizontal error bars with corresponding colors
        for i, (mean, low, high, disease) in enumerate(zip(metric_means, metric_lows, metric_highs, disease_labels)):
            plt.hlines(y=i, xmin=low, xmax=high, color=palette[i], alpha=0.7)
            plt.scatter(mean, i, color=palette[i], zorder=3)

        # Dashed line at 0.5
        plt.axvline(0.5, linestyle='dashed', color='grey', linewidth=1)

        # Formatting
        plt.yticks(range(len(disease_labels)), disease_labels)
        plt.xlim(0.4, 1.0)
        plt.xlabel(metric)
        plt.ylabel("Diseases")
        plt.title(f"{metric} with 95% Confidence Intervals")
        plt.grid(True, linestyle='dotted', alpha=0.6)

        # Save the plot
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, f"{metric.lower()}_confidence_intervals.svg"))
        plt.show()
        
    @staticmethod
    def generate_all_plots(dfs, out_dir):
        """
        Generates and saves confidence interval plots for all metrics.

        Args:
            dfs (dict): Dictionary containing DataFrames with metrics for different diseases.
            out_dir (str): Directory where plots should be saved.
        """
        for metric in ["Sensitivity", "NPV", "Specificity", "PPV", "Accuracy"]:
            ComprehensiveMulticlassROC.plot_metric_with_ci(dfs, metric, out_dir)
                
def compute_accuracy(sample, threshold, y_true_variable, independent_variable):
    """
    Computes the accuracy for a given threshold.

    Parameters:
    - sample: DataFrame with the data.
    - threshold: float with the threshold to use for classifying the scores.
    - y_true_variable: string with the name of the column containing the true binary labels.
    - independent_variable: string with the name of the column containing the independent variable (classifier scores).

    Returns:
    - Scalar with the accuracy.
    """
    y_true = sample[y_true_variable]
    scores = sample[independent_variable]
    
    predictions = [0 if score <= threshold else 1 for score in scores]
    return accuracy_score(y_true, predictions)

def compute_auc(sample, y_true_variable, independent_variable):
    """
    Computes the Area Under the Curve (AUC) for the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - sample: DataFrame with the data.
    - y_true_variable: string with the name of the column containing the true binary labels.
    - independent_variable: string with the name of the column containing the independent variable (classifier scores).

    Returns:
    - Scalar with the AUC.
    """
    y_true = sample[y_true_variable]
    scores = sample[independent_variable]

    # calculate the false positive rate and true positive rate for all thresholds of the classification
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # calculate the AUC and return it
    return auc(fpr, tpr)

def calculate_point_accuracy_and_distribution(data_df, scores, y_true, bootstrap_samples):
    # Calculate accuracies for all thresholds
    thresholds_unique = sorted(list(set(scores)))  # get unique score values and sort them
    accuracies = []

    bootstrap_results = {}
    for threshold in thresholds_unique:
        predictions = [0 if score <= threshold else 1 for score in scores]
        accuracy = accuracy_score(y_true, predictions)
        
        # Bootstrap Distribution
        func_args = {'threshold': threshold, 'y_true_variable': y_true, 'independent_variable': scores}
        bootstrap_results[f'{threshold}'] =  bootstrap_distribution_statistics(data_df, compute_accuracy, func_args, bootstrap_samples=bootstrap_samples)

    # Create a DataFrame to store thresholds and corresponding accuracies
    df_accuracies = pd.DataFrame({
        "Threshold": thresholds_unique,
        "Accuracy": accuracies
    })
    return df_accuracies, bootstrap_results

def compute_sensitivity_specificity(data, y_true_variable, independent_variable, threshold):
    y_true = data[y_true_variable]
    scores = data[independent_variable]

    # Compute predictions based on threshold
    predictions = [1 if score <= threshold else 0 for score in scores]

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    # Compute and return sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity



###----- FFunctions for Evaluation Using Above Classes -----##
'''
Hanging imports to facilitate easy transplant of code.
'''
import numpy as np
from calvin_utils.statistical_utils.classification_statistics import ComprehensiveMulticlassROC
from calvin_utils.statistical_utils.logistic_regression import LogisticRegression
import os
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm

def resample_df(data_df):
    n_samples = data_df.shape[0]
    # Shuffle the indices
    shuffled_indices = np.random.permutation(n_samples)
    # Reorder the DataFrame based on the shuffled indices but keep the original index
    shuffled_df = data_df.iloc[shuffled_indices].reset_index(drop=True).set_index(data_df.index)
    return shuffled_df

def permute_auc_difference(data_df, formula1, formula2, cal_palm, n_iterations=1000):
    auc_diffs = []
    for i in tqdm(range(n_iterations)):
        try:
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                # Define design matrices and outcome matrices for both formulas
                outcome_matrix, design_matrix1 = cal_palm.define_design_matrix(formula1, data_df)
                _, design_matrix2 = cal_palm.define_design_matrix(formula2, data_df)
                
                # Permute the outcomes
                if i == 0:
                    resampled_df = outcome_matrix
                else:
                    resampled_df = resample_df(outcome_matrix)

                # Fit the logistic regression model for the first formula
                logreg1 = LogisticRegression(resampled_df, design_matrix1)
                results1 = logreg1.run()

                # Fit the logistic regression model for the second formula
                logreg2 = LogisticRegression(resampled_df, design_matrix2)
                results2 = logreg2.run()

                # Evaluate the models
                evaluator1 = ComprehensiveMulticlassROC(fitted_model=results1, observation_df=resampled_df, normalization='true', thresholds=None, out_dir=None)
                micro_auc1 = evaluator1.get_micro_auc()

                evaluator2 = ComprehensiveMulticlassROC(fitted_model=results2, observation_df=resampled_df, normalization='true', thresholds=None, out_dir=None)
                micro_auc2 = evaluator2.get_micro_auc()

                # Store the difference in micro-average AUCs
                if i == 0:
                    obs_diff = micro_auc1 - micro_auc2
                    print(f"F1: {micro_auc1} | F2: {micro_auc2}")
                else:
                    auc_diffs.append(micro_auc1 - micro_auc2)
            
        except Exception as e:
            print(e)
            continue
    # Calculate p-value based on the distribution of differences
    auc_diffs = np.array(auc_diffs)
    p_value = np.mean(auc_diffs >= obs_diff)

    # Calculate confidence intervals for the difference
    lower_ci = np.percentile(auc_diffs, 2.5)
    upper_ci = np.percentile(auc_diffs, 97.5)
    
    return obs_diff, lower_ci, upper_ci, p_value

def bootstrap_auc(outcome_matrix, design_matrix, n_iterations=1000, model=None):
    auc_scores = []
    n_samples = outcome_matrix.shape[0]
    
    for i in tqdm(range(n_iterations)):
        # Suppress both stdout and stderr
        try:
            with open(os.devnull, 'w') as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
                # Generate a bootstrap sample
                resample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
                outcome_matrix_resampled = outcome_matrix.iloc[resample_idx]
                design_matrix_resampled = design_matrix.iloc[resample_idx]

                # Fit the logistic regression model
                if model is None:
                    logreg = LogisticRegression(outcome_matrix_resampled, design_matrix_resampled)
                    results = logreg.run()
                    test = ComprehensiveMulticlassROC(fitted_model=results, observation_df=outcome_matrix_resampled, normalization='true', thresholds=None, out_dir=None)
                else:
                    results = model.predict(design_matrix_resampled)
                    test = ComprehensiveMulticlassROC(fitted_model=None, predictions_df=results, observation_df=outcome_matrix_resampled, normalization='true', thresholds=None, out_dir=None)
                # Evaluate the model
                micro_auc = test.get_micro_auc()
                auc_scores.append(micro_auc)
        except:
            continue
    # Calculate confidence intervals
    lower_ci = np.percentile(auc_scores, 2.5)
    upper_ci = np.percentile(auc_scores, 97.5)    
    return np.mean(auc_scores), lower_ci, upper_ci


### Specific Functions ###
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd

def calculate_youden(
    raw_observations: np.ndarray,
    raw_predictions: np.ndarray,
    outcome_matrix_cols
):
    """
    Calculates Youden's J for each class, and bootstraps Sensitivity, Specificity, NPV, PPV, and Accuracy
    at the corresponding cut point. Returns a summary DataFrame for each class.

    Parameters
    ----------
    raw_observations : np.ndarray
        Ground truth one-hot encoded array of shape (n_samples, n_classes).
    raw_predictions : np.ndarray
        Predicted probabilities for each class of shape (n_samples, n_classes).
    outcome_matrix_cols : list-like
        The column (class) labels, typically from `outcome_matrix.columns`.

    Returns
    -------
    metrics_dfs : dict
        two dictionaries where keys are class names are keys. values for first are youden's j, values for second are threhsold for each youdens j
    """
    n_classes = raw_observations.shape[1]

    youden_dict = {}
    threshold_dict = {}
    for i in range(n_classes):
        class_name = outcome_matrix_cols[i]
        y_true = raw_observations[:, i]
        y_score = raw_predictions[:, i]

        # Calculate the optimal threshold using Youden's J
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]

        # Store the summary DataFrame for the current class
        youden_dict[class_name] = youden_j[optimal_idx]
        threshold_dict[class_name] = optimal_threshold
    return youden_dict, threshold_dict

def calculate_metrics_at_threshold(
    raw_observations: np.ndarray,
    raw_predictions: np.ndarray,
    outcome_matrix_cols,
    threshold_dict,
    n_bootstraps: int = 1000,
    random_state: int = None,
    ci_alpha: float = 0.95
):
    """
    Calculates Youden's J for each class, and bootstraps Sensitivity, Specificity, NPV, PPV, and Accuracy
    at the corresponding cut point. Returns a summary DataFrame for each class.

    Parameters
    ----------
    raw_observations : np.ndarray
        Ground truth one-hot encoded array of shape (n_samples, n_classes).
    raw_predictions : np.ndarray
        Predicted probabilities for each class of shape (n_samples, n_classes).
    outcome_matrix_cols : list-like
        The column (class) labels, typically from `outcome_matrix.columns`.
    threshold_dict : dict
        the dictionary where keys correspond to classes and values correspond to thresholds to evaluate 
    n_bootstraps : int
        Number of bootstrap iterations.
    random_state : int
        Controls reproducibility of the random sampling.
    ci_alpha : float
        Confidence interval coverage (0 < ci_alpha < 1). 0.95 -> 95% CI.

    Returns
    -------
    metrics_dfs : dict
        A dictionary where keys are class names and values are DataFrames with rows
        as metrics (Sensitivity, Specificity, NPV, PPV, Accuracy) and columns as
        mean, lower bound, and upper bound.
    """
    rng = np.random.default_rng(seed=random_state)
    n_classes = raw_observations.shape[1]

    metrics_dfs = {}
    for i in range(n_classes):
        class_name = outcome_matrix_cols[i]
        y_true = raw_observations[:, i]
        y_score = raw_predictions[:, i]
        optimal_threshold = threshold_dict[class_name]

        # Call a helper function to calculate bootstrapped metrics
        metrics_summary = bootstrap_metrics_at_cutpoint(y_true, y_score, optimal_threshold, n_bootstraps, rng, ci_alpha)

        # Store the summary DataFrame for the current class
        metrics_dfs[class_name] = metrics_summary
    return metrics_dfs

def calculate_youden_and_metrics(
    raw_observations: np.ndarray,
    raw_predictions: np.ndarray,
    outcome_matrix_cols,
    n_bootstraps: int = 1000,
    random_state: int = None,
    ci_alpha: float = 0.95,
    out_dir: None = None
):
    """
    Calculates Youden's J for each class, and bootstraps Sensitivity, Specificity, NPV, PPV, and Accuracy
    at the corresponding cut point. Returns a summary DataFrame for each class.

    Parameters
    ----------
    raw_observations : np.ndarray
        Ground truth one-hot encoded array of shape (n_samples, n_classes).
    raw_predictions : np.ndarray
        Predicted probabilities for each class of shape (n_samples, n_classes).
    outcome_matrix_cols : list-like
        The column (class) labels, typically from `outcome_matrix.columns`.
    n_bootstraps : int
        Number of bootstrap iterations.
    random_state : int
        Controls reproducibility of the random sampling.
    ci_alpha : float
        Confidence interval coverage (0 < ci_alpha < 1). 0.95 -> 95% CI.

    Returns
    -------
    metrics_dfs : dict
        A dictionary where keys are class names and values are DataFrames with rows
        as metrics (Sensitivity, Specificity, NPV, PPV, Accuracy) and columns as
        mean, lower bound, and upper bound.
    """
    rng = np.random.default_rng(seed=random_state)
    n_classes = raw_observations.shape[1]

    youden_dict = {}
    metrics_dfs = {}
    print("--Optimal Threshold--")
    for i in range(n_classes):
        class_name = outcome_matrix_cols[i]
        y_true = raw_observations[:, i]
        y_score = raw_predictions[:, i]

        # Calculate the optimal threshold using Youden's J
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]

        cm = get_confusion_matrix(y_true, y_score, optimal_threshold, class_name, normalization='pred', out_dir=out_dir)
        # Call a helper function to calculate bootstrapped metrics
        metrics_summary = bootstrap_metrics_at_cutpoint(
            y_true, y_score, optimal_threshold, n_bootstraps, rng, ci_alpha
        )
        youden_dict[class_name] = [{'threshold': youden_j[optimal_idx]}, {'c_matrix': cm}]
        print(f"{class_name}: {optimal_threshold}")

        # Store the summary DataFrame for the current class
        metrics_dfs[class_name] = metrics_summary
    return metrics_dfs, youden_dict

def get_confusion_matrix(y_true, y_score, optimal_threshold, class_name, normalization, out_dir=None, verbose=True, **kwargs):
    '''
    Params:
    normalization - str, default='pred'
        - 'pred' : normalize by the number of predictions (default)
        - 'true' : normalize by the number of true labels
        - 'all' : normalize by the number of samples
        - None : no normalization
    out_dir - str, default=None
        - directory to save the confusion matrix plot
    '''
    # onyl works for binomial case (meant for a OVR confusion matrix )
    y_pred = (y_score >= optimal_threshold).astype(int)
    # Plot the confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=normalization)
    if verbose:
        conf_matrix2 = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=None)
        print(f"True Positives: {conf_matrix2[1, 1]}")
        print(f"True Negatives: {conf_matrix2[0, 0]}")
        print(f"False Positives: {conf_matrix2[0, 1]}")
        print(f"False Negatives: {conf_matrix2[1, 0]}")
    plot_cm(conf_matrix, class_name, out_dir=out_dir, **kwargs)
    return conf_matrix

def plot_cm(cm, class_name, cmap='viridis', annot=True, normalization=None, out_dir=None):
    labels = np.array([ f"Not {class_name}", f"Is {class_name}"])
    if normalization is None: digit_fmt = '1g'
    else: digit_fmt = '.2f'
    plt.figure(figsize=(8, 6))
    simple_heatmap(
        cm,
        dataset_name="",
        palette=cmap,
        annot=annot,
        fmt=digit_fmt,
        labels=labels,
    )
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix {class_name}')
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{class_name}_optimal_confusion_matrix.svg'))
    plt.show()
        
def bootstrap_metrics_at_cutpoint(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cutpoint: float,
    n_bootstraps: int,
    rng,
    ci_alpha: float
):
    """
    Bootstraps Sensitivity, Specificity, NPV, PPV, and Accuracy at a given cut point.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth binary labels.
    y_score : np.ndarray
        Predicted probabilities for the positive class.
    cutpoint : float
        The threshold at which to calculate the metrics.
    n_bootstraps : int
        Number of bootstrap iterations.
    rng : np.random.Generator
        Random number generator for reproducibility.
    ci_alpha : float
        Confidence interval coverage (0 < ci_alpha < 1). 0.95 -> 95% CI.

    Returns
    -------
    metrics_summary_df : pd.DataFrame
        DataFrame with rows as metrics (Sensitivity, Specificity, NPV, PPV, Accuracy)
        and columns as mean, lower bound, and upper bound.
    """
    metrics = {"Sensitivity": [], "Specificity": [], "NPV": [], "PPV": [], "Accuracy": []}

    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        # Sample with replacement
        idx = rng.integers(0, n_samples, n_samples)
        y_true_boot = y_true[idx]
        y_score_boot = y_score[idx]

        # Convert probabilities to binary predictions using the cutpoint
        y_pred_boot = (y_score_boot >= cutpoint).astype(int)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_boot, y_pred_boot).ravel()

        # Calculate metrics
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # Sensitivity
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan  # Specificity
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan  # Negative Predictive Value
        acc = accuracy_score(y_true_boot, y_pred_boot)     # Accuracy

        metrics["Sensitivity"].append(sens)
        metrics["Specificity"].append(spec)
        metrics["NPV"].append(npv)
        metrics["PPV"].append(ppv)
        metrics["Accuracy"].append(acc)

    # Calculate mean, lower bound, and upper bound for each metric
    rows = []
    alpha_lower = (1.0 - ci_alpha) / 2.0
    alpha_upper = 1.0 - alpha_lower

    for metric, values in metrics.items():
        values_clean = [v for v in values if not np.isnan(v)]
        if len(values_clean) == 0:
            rows.append([metric, np.nan, np.nan, np.nan])
        else:
            mean_val = np.mean(values_clean)
            lower_bound = np.quantile(values_clean, alpha_lower)
            upper_bound = np.quantile(values_clean, alpha_upper)
            rows.append([metric, mean_val, lower_bound, upper_bound])

    metrics_summary_df = pd.DataFrame(
        rows, columns=["Metric", "Mean", f"{alpha_lower:.3g}%", f"{alpha_upper:.3g}%"]
    ).set_index("Metric")

    return metrics_summary_df

def save_dfs(dfs:dict, out_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    for name, df in dfs.items():
        if out_dir is not None:
            df.to_csv(out_dir + '/' + name + '.csv')
