import argparse
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def main(training_data_file, testing_data_file, k):
    '''Method for preprocessing, building knn model, and running inference on knn model.

    Arguments:
        training_data_file:
        testing_data_file:
        k:
    '''
    with open(training_data_file, 'r') as f:
        training_data = json.load(f)
    with open(testing_data_file, 'r') as f:
        testing_data = json.load(f)
    cleaned_testing_data = preprocess_data_for_knn(testing_data)
    cleaned_training_data = preprocess_data_for_knn(training_data)
    knn_x, knn_y = build_knn_model(cleaned_training_data, int(k))
    
    # Calculate accuracy for error bounds 0 through 5
    accuracies = []
    for error_bound in range(6):
        accuracy = test_knn_model(knn_x, knn_y, cleaned_testing_data, error_bound)
        accuracies.append(accuracy)
    
    # Generate the accuracy graph for each error bound
    generate_accuracy_graph(accuracies)

def test_knn_model(knn_x, knn_y, test_set, error_bound):
    num_correct = 0

    for num, sample in enumerate(test_set):
        x_label = sample['x']
        y_label = sample['y']
        feature_vector = []
        for key, value in sample.items():
            if key != 'x' and key != 'y':
                feature_vector.append(value)

        x_predict = knn_x.predict([feature_vector])
        y_predict = knn_y.predict([feature_vector])
        x_error = abs(x_predict[0] - x_label)
        y_error = abs(y_predict[0] - y_label)

        # Check if the error is within the given error bound
        if x_error <= error_bound and y_error <= error_bound:
            num_correct += 1

    accuracy_as_percent = num_correct / len(test_set) * 100
    print(f"Accuracy with error bound {error_bound}: {accuracy_as_percent:.2f}%")
    return accuracy_as_percent

def preprocess_data_for_knn(crowd_sourced_data):
    '''Preprocess the crowd_sourced data'''
    # Collect all unique phone keys
    all_keys = set()
    for sample in crowd_sourced_data:
        all_keys.update(sample.keys())
    
    # Remove 'x' and 'y' as they aren't features
    all_keys.discard('x')
    all_keys.discard('y')
    
    sorted_keys = sorted(all_keys)  # Sort keys alphabetically for consistency

    processed_data = []
    for sample in crowd_sourced_data:
        processed_sample = {
            'x': sample['x'],
            'y': sample['y']
        }
        # Add each phone's signal strength or -1 if not present
        for key in sorted_keys:
            processed_sample[key] = sample.get(key, -1)
        processed_data.append(processed_sample)

    return processed_data

def build_knn_model(processed_data, k):
    '''Build knn regression model that predicts location'''
    knn_x = KNeighborsClassifier(n_neighbors=k)
    knn_y = KNeighborsClassifier(n_neighbors=k)
    x_labels = []
    y_labels = []
    feature_vectors = []
    for sample in processed_data:
        x_labels.append(sample['x'])
        y_labels.append(sample['y'])
        feature_vector = []
        for key, value in sample.items():
            if key != 'x' and key != 'y':
                feature_vector.append(value)
        feature_vectors.append(feature_vector)
    knn_x.fit(feature_vectors, x_labels)
    knn_y.fit(feature_vectors, y_labels)
    return knn_x, knn_y

def generate_accuracy_graph(accuracies):
    '''Generate a graph of accuracy over error bounds'''
    error_bounds = list(range(len(accuracies)))
    plt.plot(error_bounds, accuracies, marker='o')
    plt.title('KNN Model Accuracy at Different Error Bounds')
    plt.xlabel('Error Bound')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.xticks(error_bounds)  # Show integer ticks for error bounds
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--training_data_file', default='./training-data.json',
                        help='Path to training data in JSON format.')
    parser.add_argument('--testing_data_file', default='./testing-data.json',
                        help='Path to testing data in JSON format.')
    parser.add_argument('--k', default=5,
                        help='Number of Nearest Neighbors used in prediction')
    args = parser.parse_args()
    main(args.training_data_file, args.testing_data_file, args.k)
