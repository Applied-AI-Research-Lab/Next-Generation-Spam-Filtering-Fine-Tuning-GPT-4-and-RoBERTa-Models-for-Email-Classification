import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import seaborn as sns


class EvaluationMethods:
    def __init__(self, dataset_path=''):
        self.dataset_path = dataset_path
        self.pre_path = '../Dataset_1/'

    def evaluate_results(self, original, prediction, model_name):
        data = pd.read_csv(self.pre_path + self.dataset_path)
        accuracy = round(accuracy_score(data[original], data[prediction]), 4)
        precision = round(precision_score(data[original], data[prediction], average='weighted'), 4)
        recall = round(recall_score(data[original], data[prediction], average='weighted'), 4)
        f1 = round(f1_score(data[original], data[prediction], average='weighted'), 4)

        # Create a DataFrame with the evaluation results including the 'model' column
        evaluation_df = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1': [f1]
        })

        # Append the results to the existing CSV file or create a new one
        evaluation_df.to_csv(self.pre_path + 'evaluation-results.csv', mode='a',
                             header=not os.path.exists(self.pre_path + 'evaluation-results.csv'), index=False)

        return {'Model': model_name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}

    def scatterplot(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)
        prediction = df[prediction_column]
        original = df[original_column]

        # Calculate Mean Absolute Error
        mae = abs(original - prediction).mean()

        # Create a scatter plot with a regression line
        sns.regplot(x=original, y=prediction, scatter_kws={'alpha': 0.5})

        plt.xlabel(original_column)
        plt.ylabel(prediction_column)

        # Save the scatterplot image to the Datasets folder
        plt.savefig(os.path.join(self.pre_path + 'Plots/', prediction_column + '.png'))

        # Show the plot
        plt.show()

        return mae

    def plot_confusion_matrix(self, original_column, prediction_column):
        dataframe = pd.read_csv(self.pre_path + self.dataset_path)

        # Extract data from DataFrame
        y_true = dataframe[original_column]
        y_pred = dataframe[prediction_column]

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix \n('+prediction_column+')')
        plt.show()

    def count_matching_rows(self, original_column, prediction_column):
        df = pd.read_csv(self.pre_path + self.dataset_path)

        # Count the number of same value rows
        matching_rows = df[df[original_column] == df[prediction_column]]

        return len(matching_rows)


# Example Usage
# Instantiate the DatasetMethods class by providing the (dataset_path)
EVM = EvaluationMethods(dataset_path='test_set.csv')

# # Count correct predictions
# print(str(EVM.count_matching_rows('Spam', 'gpt_bm_prediction')))
# print(str(EVM.count_matching_rows('Spam', 'gpt_ft_prediction')))
# print(str(EVM.count_matching_rows('Spam', 'roberta_optimizer_Adam_lr_2e-05_epochs_3_bs_6_maxlen_512_prediction')))
# print(str(EVM.count_matching_rows('Spam', 'bert_optimizer_Adam_lr_2e-05_epochs_3_bs_6_maxlen_512_prediction')))
# print(str(EVM.count_matching_rows('Spam', 'cnn_optimizer_Adam_lr_2e-05_epochs_3_bs_6_maxlen_4096_prediction')))

# # Evaluate the predictions made by each model
# print(f'base:gpt-4-0125-preview: ' + str(EVM.evaluate_results('Spam', 'gpt_bm_prediction', 'base:gpt-4-0125-preview')))
# print(f'ft:gpt-4: ' + str(EVM.evaluate_results('Spam', 'gpt_ft_prediction', 'ft:gpt-4')))
# print(f'ft:roberta-adam: ' + str(EVM.evaluate_results('Spam', 'ft_roberta_adam', 'ft:roberta-adam')))
# print(f'ft:bert-adam: ' + str(EVM.evaluate_results('Spam', 'ft_bert_adam', 'ft:bert-adam')))
# print(f'ft:cnn-adam: ' + str(EVM.evaluate_results('Spam', 'ft_cnn_adam', 'ft:cnn-adam')))
# print(f'ft:gpt-4-cross: ' + str(EVM.evaluate_results('Spam', 'gpt_ft_cross', 'ft:gpt-4-cross')))
# print(f'ft:roberta-adam-cross: ' + str(EVM.evaluate_results('Spam', 'ft_roberta_adam_cross', 'ft:roberta-adam-cross')))
# print(f'ft:bert-adam-cross: ' + str(EVM.evaluate_results('Spam', 'ft_bert_adam_cross', 'ft:bert-adam-cross')))
# print(f'ft:cnn-adam-cross: ' + str(EVM.evaluate_results('Spam', 'ft_cnn_adam_cross', 'ft:cnn-adam-cross')))

# Create scatterplots
# print(EVM.scatterplot(original_column='Spam', prediction_column='gpt_bm_prediction'))
# print(EVM.scatterplot(original_column='Spam', prediction_column='gpt_ft_prediction'))
# print(EVM.scatterplot(original_column='Spam', prediction_column='ft_roberta_adam'))
# print(EVM.scatterplot(original_column='Spam', prediction_column='ft_bert_adam'))
# print(EVM.scatterplot(original_column='Spam', prediction_column='ft_cnn_adam'))

# Create confusion matrix
# EVM.plot_confusion_matrix(original_column='Spam', prediction_column='gpt_bm_prediction')
# EVM.plot_confusion_matrix(original_column='Spam', prediction_column='gpt_ft_prediction')
# EVM.plot_confusion_matrix(original_column='Spam', prediction_column='ft_roberta_adam')
# EVM.plot_confusion_matrix(original_column='Spam', prediction_column='ft_bert_adam')
# EVM.plot_confusion_matrix(original_column='Spam', prediction_column='ft_cnn_adam')


