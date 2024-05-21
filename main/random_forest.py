from sklearn.ensemble import RandomForestClassifier
import feature_extractor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, train_test_split

PLOT_ENABLED = True

BIAX_TENSION = 'Biax Tension'
PLANAR_COMPRESSION = 'Planar Compression'
PLANAR_TENSION = 'Planar Tension'
SIMPLE_SHEAR = 'Simple Shear'
UNIAX_COMPRESSION = 'Uniax Compression'
UNIAX_TENSION = 'Uniax Tension'

BIAX_TENSION_PREDICTION = 'Biax Tension Prediction'
PLANAR_COMPRESSION_PREDICTION = 'Planar Compression Prediction'
PLANAR_TENSION_PREDICTION = 'Planar Tension Prediction'
SIMPLE_SHEAR_PREDICTION = 'Simple Shear Prediction'
UNIAX_COMPRESSION_PREDICTION = 'Uniax Compression Prediction'
UNIAX_TENSION_PREDICTION = 'Uniax Tension Prediction'

DATA_POINT = 'data_point'

NH_VOTES = 'nh_votes'
MR_VOTES = 'mr_votes'
OG_VOTES = 'og_votes'
PREDICTED_CLASS = 'predicted_class'

UNTIL_SECOND_UNDERSCORE_PATTERN = r'^[^_]*_[^_]*_'

# function to read inputs for stress type
def read_inputs(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data

# function to train a decision tree classifier with cross validation
def get_decision_tree_cross_val_score(X_train, y_train, max_depth):
    decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=1)
    scores = cross_val_score(decision_tree, X_train, y_train, cv=10, scoring='accuracy')
    return scores.mean()

# function to train decision tree classifiers with max_depths from 3 to 10
def find_best_max_depth(X_train, y_train, range, stress_type):
    max_depths = range
    scores = {}
    for max_depth in max_depths:
        score = get_decision_tree_cross_val_score(X_train, y_train, max_depth)
        scores[max_depth] = score

    # plot the scores to find the best max_depth
    if PLOT_ENABLED:
        plt.plot(scores.keys(), scores.values())
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.title('Decision Tree Accuracy for Different Max Depths - ' + stress_type)
        plt.show()

    best_max_depth = max(scores, key=scores.get)
    print(f'Best max_depth for {stress_type}:', best_max_depth)
    return best_max_depth

# function to train a random forest classifier with cross validation
def get_random_forest_cross_val_score(X_train, y_train, n_estimators, max_depth):
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)
    scores = cross_val_score(random_forest, X_train, y_train, cv=10, scoring='accuracy')
    return scores.mean()

# function to train random forest classifiers with n_estimators from 10 to 100
def find_best_n_estimators(X_train, y_train, max_depth, range, stress_type):
    n_estimators = range
    scores = {}
    for n_estimator in n_estimators:
        score = get_random_forest_cross_val_score(X_train, y_train, n_estimator, max_depth)
        scores[n_estimator] = score

    # plot the scores to find the best n_estimators
    if PLOT_ENABLED:
        plt.plot(scores.keys(), scores.values())
        plt.xlabel('Number of Estimators')
        plt.ylabel('Accuracy')
        plt.title('Random Forest Accuracy for Different Number of Estimators - ' + stress_type)
        plt.show()

    best_n_estimator = max(scores, key=scores.get)
    print(f'Best n_estimator for {stress_type}:', best_n_estimator)
    return best_n_estimator

# function to train random forest classifier and return it
def train_random_forest(X_train, y_train, n_estimators, max_depth):
    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=1)
    random_forest.fit(X_train, y_train)
    return random_forest

def evaluate_model(random_forest: RandomForestClassifier, X_test, y_test, stress_type):
    y_pred = random_forest.predict(X_test)
    accuracy = random_forest.score(X_test, y_test)
    print(f'Accuracy for {stress_type}:', accuracy)

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, y_pred)
    # use heatmap to plot confusion matrix
    if PLOT_ENABLED:
        sns.heatmap(confusion_matrix, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {stress_type}')
        plt.show()

def classify_biax_tension():
    df_biax_tension = read_inputs('./extracted_features_data/biax_tension_features.csv')
    df_biax_tension[DATA_POINT] = df_biax_tension[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_biax_tension_test = df_biax_tension[df_biax_tension[DATA_POINT] % 5 == 0]
    df_biax_tension_train = df_biax_tension[df_biax_tension[DATA_POINT] % 5 != 0]
    X_train = df_biax_tension_train.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    y_train = df_biax_tension_train[feature_extractor.CLASS]
    X_test = df_biax_tension_test.drop([feature_extractor.SAMPLE_NAME,feature_extractor.CLASS, DATA_POINT], axis=1)
    y_test = df_biax_tension_test[feature_extractor.CLASS]

    # find the best max_depth for decision_tree
    max_depth = find_best_max_depth(X_train, y_train, range(3,11), BIAX_TENSION)
    best_max_depth = max_depth

    # find the best n_estimators for random_forest
    n_estimators = find_best_n_estimators(X_train, y_train, best_max_depth, range(3, 19, 3), BIAX_TENSION)
    best_n_estimators = n_estimators

    # train random forest classifier
    random_forest = train_random_forest(X_train, y_train, best_n_estimators, best_max_depth)
    evaluate_model(random_forest, X_test, y_test, BIAX_TENSION)

    return random_forest

def classify_planar_compression():
    df_planar_compression = read_inputs('./extracted_features_data/planar_compression_features.csv')
    df_planar_compression[DATA_POINT] = df_planar_compression[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_planar_compression_test = df_planar_compression[df_planar_compression[DATA_POINT] % 5 == 0]
    df_planar_compression_train = df_planar_compression[df_planar_compression[DATA_POINT] % 5 != 0]
    X_train = df_planar_compression_train.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    y_train = df_planar_compression_train[feature_extractor.CLASS]
    X_test = df_planar_compression_test.drop([feature_extractor.SAMPLE_NAME,feature_extractor.CLASS, DATA_POINT], axis=1)
    y_test = df_planar_compression_test[feature_extractor.CLASS]

    # find the best max_depth for decision_tree
    max_depth = find_best_max_depth(X_train, y_train, range(3,13), PLANAR_COMPRESSION)
    best_max_depth = max_depth

    # find the best n_estimators for random_forest
    n_estimators = find_best_n_estimators(X_train, y_train, best_max_depth, range(15, 30, 3), PLANAR_COMPRESSION)
    best_n_estimators = n_estimators

    # train random forest classifier
    random_forest = train_random_forest(X_train, y_train, best_n_estimators, best_max_depth)
    evaluate_model(random_forest, X_test, y_test, PLANAR_COMPRESSION)

    return random_forest

def classify_planar_tension():
    df_planar_tension = read_inputs('./extracted_features_data/planar_tension_features.csv')
    df_planar_tension[DATA_POINT] = df_planar_tension[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_planar_tension_test = df_planar_tension[df_planar_tension[DATA_POINT] % 5 == 0]
    df_planar_tension_train = df_planar_tension[df_planar_tension[DATA_POINT] % 5 != 0]
    X_train = df_planar_tension_train.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    y_train = df_planar_tension_train[feature_extractor.CLASS]
    X_test = df_planar_tension_test.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    y_test = df_planar_tension_test[feature_extractor.CLASS]

    # find the best max_depth for decision_tree
    max_depth = find_best_max_depth(X_train, y_train, range(3,13), PLANAR_TENSION)
    best_max_depth = max_depth

    # find the best n_estimators for random_forest
    n_estimators = find_best_n_estimators(X_train, y_train, best_max_depth, range(40, 76, 5), PLANAR_TENSION)
    best_n_estimators = n_estimators

    # train random forest classifier
    random_forest = train_random_forest(X_train, y_train, best_n_estimators, best_max_depth)
    evaluate_model(random_forest, X_test, y_test, PLANAR_TENSION)

    return random_forest

def classify_simple_shear():
    df_simple_shear = read_inputs('./extracted_features_data/simple_shear_features.csv')
    df_simple_shear[DATA_POINT] = df_simple_shear[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_simple_shear_test = df_simple_shear[df_simple_shear[DATA_POINT] % 5 == 0]
    df_simple_shear_train = df_simple_shear[df_simple_shear[DATA_POINT] % 5 != 0]
    X_train = df_simple_shear_train.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    y_train = df_simple_shear_train[feature_extractor.CLASS]
    X_test = df_simple_shear_test.drop([feature_extractor.SAMPLE_NAME,feature_extractor.CLASS, DATA_POINT], axis=1)
    y_test = df_simple_shear_test[feature_extractor.CLASS]

    # find the best max_depth for decision_tree
    max_depth = find_best_max_depth(X_train, y_train, range(3, 10), SIMPLE_SHEAR)
    best_max_depth = max_depth

    # find the best n_estimators for random_forest
    n_estimators = find_best_n_estimators(X_train, y_train, best_max_depth, range(5, 31, 5), SIMPLE_SHEAR)
    best_n_estimators = n_estimators

    # train random forest classifier
    random_forest = train_random_forest(X_train, y_train, best_n_estimators, best_max_depth)
    evaluate_model(random_forest, X_test, y_test, SIMPLE_SHEAR)

    return random_forest

def classify_uniax_compression():
    df_uniax_compression = read_inputs('./extracted_features_data/uniax_compression_features.csv')
    df_uniax_compression[DATA_POINT] = df_uniax_compression[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_uniax_compression_test = df_uniax_compression[df_uniax_compression[DATA_POINT] % 5 == 0]
    df_uniax_compression_train = df_uniax_compression[df_uniax_compression[DATA_POINT] % 5 != 0]
    X_train = df_uniax_compression_train.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    y_train = df_uniax_compression_train[feature_extractor.CLASS]
    X_test = df_uniax_compression_test.drop([feature_extractor.SAMPLE_NAME,feature_extractor.CLASS, DATA_POINT], axis=1)
    y_test = df_uniax_compression_test[feature_extractor.CLASS]

    # find the best max_depth for decision_tree
    max_depth = find_best_max_depth(X_train, y_train, range(3, 13), UNIAX_COMPRESSION)
    best_max_depth = max_depth

    # find the best n_estimators for random_forest
    n_estimators = find_best_n_estimators(X_train, y_train, best_max_depth, range(26, 52, 3), UNIAX_COMPRESSION)
    best_n_estimators = n_estimators

    # train random forest classifier
    random_forest = train_random_forest(X_train, y_train, best_n_estimators, best_max_depth)
    evaluate_model(random_forest, X_test, y_test, UNIAX_COMPRESSION)

    return random_forest

def classify_uniax_tension():
    df_uniax_tension = read_inputs('./extracted_features_data/uniax_tension_features.csv')
    df_uniax_tension[DATA_POINT] = df_uniax_tension[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_uniax_tension_test = df_uniax_tension[df_uniax_tension[DATA_POINT] % 5 == 0]
    df_uniax_tension_train = df_uniax_tension[df_uniax_tension[DATA_POINT] % 5 != 0]
    X_train = df_uniax_tension_train.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    y_train = df_uniax_tension_train[feature_extractor.CLASS]
    X_test = df_uniax_tension_test.drop([feature_extractor.SAMPLE_NAME,feature_extractor.CLASS, DATA_POINT], axis=1)
    y_test = df_uniax_tension_test[feature_extractor.CLASS]

    # find the best max_depth for decision_tree
    max_depth = find_best_max_depth(X_train, y_train, range(2, 13), UNIAX_TENSION)
    best_max_depth = max_depth

    # find the best n_estimators for random_forest
    n_estimators = find_best_n_estimators(X_train, y_train, best_max_depth, range(25, 71, 5), UNIAX_TENSION)
    best_n_estimators = n_estimators

    # train random forest classifier
    random_forest = train_random_forest(X_train, y_train, best_n_estimators, best_max_depth)
    evaluate_model(random_forest, X_test, y_test, UNIAX_TENSION)

    return random_forest

def get_biax_tension_test_predictions(random_forest: RandomForestClassifier):
    df_biax_tension = read_inputs('./extracted_features_data/biax_tension_features.csv')
    # remove subrstring before the second underscore in the sample name
    df_biax_tension[feature_extractor.SAMPLE_NAME] = df_biax_tension[feature_extractor.SAMPLE_NAME].str.replace(UNTIL_SECOND_UNDERSCORE_PATTERN, '', regex=True)
    df_biax_tension[DATA_POINT] = df_biax_tension[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_biax_tension_test = df_biax_tension[df_biax_tension[DATA_POINT] % 5 == 0]
    X_test = df_biax_tension_test.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    
    y_test_pred = random_forest.predict(X_test)
    df_biax_tension_test[BIAX_TENSION_PREDICTION] = y_test_pred
    df_biax_tension_test_results = df_biax_tension_test[[feature_extractor.SAMPLE_NAME, BIAX_TENSION_PREDICTION]]
    return df_biax_tension_test_results

def get_planar_compression_test_predictions(random_forest: RandomForestClassifier):
    df_planar_compression = read_inputs('./extracted_features_data/planar_compression_features.csv')
    # remove subrstring before the second underscore in the sample name
    df_planar_compression[feature_extractor.SAMPLE_NAME] = df_planar_compression[feature_extractor.SAMPLE_NAME].str.replace(UNTIL_SECOND_UNDERSCORE_PATTERN, '', regex=True)
    df_planar_compression[DATA_POINT] = df_planar_compression[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_planar_compression_test = df_planar_compression[df_planar_compression[DATA_POINT] % 5 == 0]
    X_test = df_planar_compression_test.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    
    y_test_pred = random_forest.predict(X_test)
    df_planar_compression_test[PLANAR_COMPRESSION_PREDICTION] = y_test_pred
    df_planar_compression_test_results = df_planar_compression_test[[feature_extractor.SAMPLE_NAME, PLANAR_COMPRESSION_PREDICTION]]
    return df_planar_compression_test_results

def get_planar_tension_test_predictions(random_forest: RandomForestClassifier):
    df_planar_tension = read_inputs('./extracted_features_data/planar_tension_features.csv')
    # remove subrstring before the second underscore in the sample name
    df_planar_tension[feature_extractor.SAMPLE_NAME] = df_planar_tension[feature_extractor.SAMPLE_NAME].str.replace(UNTIL_SECOND_UNDERSCORE_PATTERN, '', regex=True)
    df_planar_tension[DATA_POINT] = df_planar_tension[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_planar_tension_test = df_planar_tension[df_planar_tension[DATA_POINT] % 5 == 0]
    X_test = df_planar_tension_test.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    
    y_test_pred = random_forest.predict(X_test)
    df_planar_tension_test[PLANAR_TENSION_PREDICTION] = y_test_pred
    df_planar_tension_test_results = df_planar_tension_test[[feature_extractor.SAMPLE_NAME, PLANAR_TENSION_PREDICTION]]
    return df_planar_tension_test_results

def get_simple_shear_test_predictions(random_forest: RandomForestClassifier):
    df_simple_shear = read_inputs('./extracted_features_data/simple_shear_features.csv')
    # remove subrstring before the second underscore in the sample name
    df_simple_shear[feature_extractor.SAMPLE_NAME] = df_simple_shear[feature_extractor.SAMPLE_NAME].str.replace(UNTIL_SECOND_UNDERSCORE_PATTERN, '', regex=True)
    df_simple_shear[DATA_POINT] = df_simple_shear[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_simple_shear_test = df_simple_shear[df_simple_shear[DATA_POINT] % 5 == 0]
    X_test = df_simple_shear_test.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    
    y_test_pred = random_forest.predict(X_test)
    df_simple_shear_test[SIMPLE_SHEAR_PREDICTION] = y_test_pred
    df_simple_shear_test_results = df_simple_shear_test[[feature_extractor.SAMPLE_NAME, SIMPLE_SHEAR_PREDICTION]]
    return df_simple_shear_test_results

def get_uniax_compression_test_predictions(random_forest: RandomForestClassifier):
    df_uniax_compression = read_inputs('./extracted_features_data/uniax_compression_features.csv')
    # remove subrstring before the second underscore in the sample name
    df_uniax_compression[feature_extractor.SAMPLE_NAME] = df_uniax_compression[feature_extractor.SAMPLE_NAME].str.replace(UNTIL_SECOND_UNDERSCORE_PATTERN, '', regex=True)
    df_uniax_compression[DATA_POINT] = df_uniax_compression[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_uniax_compression_test = df_uniax_compression[df_uniax_compression[DATA_POINT] % 5 == 0]
    X_test = df_uniax_compression_test.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    
    y_test_pred = random_forest.predict(X_test)
    df_uniax_compression_test[UNIAX_COMPRESSION_PREDICTION] = y_test_pred
    df_uniax_compression_test_results = df_uniax_compression_test[[feature_extractor.SAMPLE_NAME, UNIAX_COMPRESSION_PREDICTION]]
    return df_uniax_compression_test_results

def get_uniax_tension_test_predictions(random_forest: RandomForestClassifier):
    df_uniax_tension = read_inputs('./extracted_features_data/uniax_tension_features.csv')
    # remove subrstring before the second underscore in the sample name
    df_uniax_tension[feature_extractor.SAMPLE_NAME] = df_uniax_tension[feature_extractor.SAMPLE_NAME].str.replace(UNTIL_SECOND_UNDERSCORE_PATTERN, '', regex=True)
    df_uniax_tension[DATA_POINT] = df_uniax_tension[feature_extractor.SAMPLE_NAME].str.extract('(\d+)').astype(int)
    df_uniax_tension_test = df_uniax_tension[df_uniax_tension[DATA_POINT] % 5 == 0]
    X_test = df_uniax_tension_test.drop([feature_extractor.SAMPLE_NAME, feature_extractor.CLASS, DATA_POINT], axis=1)
    
    y_test_pred = random_forest.predict(X_test)
    df_uniax_tension_test[UNIAX_TENSION_PREDICTION] = y_test_pred
    df_uniax_tension_test_results = df_uniax_tension_test[[feature_extractor.SAMPLE_NAME, UNIAX_TENSION_PREDICTION]]
    return df_uniax_tension_test_results

def summarize_votes(df_combined):
    # count the votes for each stress type
    df_combined[NH_VOTES] = df_combined.apply(lambda x: sum(x == 'nh'), axis=1)
    df_combined[MR_VOTES] = df_combined.apply(lambda x: sum(x == 'mr'), axis=1)
    df_combined[OG_VOTES] = df_combined.apply(lambda x: sum(x == 'og'), axis=1)
    df_combined[PREDICTED_CLASS] = df_combined[[NH_VOTES, MR_VOTES, OG_VOTES]].idxmax(axis=1)
    df_combined[PREDICTED_CLASS] = df_combined[PREDICTED_CLASS].str.replace('_votes', '')

    # determine class name (nh, mr, og) based on the sample name using feature_extractor.get_class
    df_combined[feature_extractor.CLASS] = df_combined[feature_extractor.SAMPLE_NAME].apply(feature_extractor.get_class)

def evaluate_combined_model(df_combined):
    y_test = df_combined[feature_extractor.CLASS]
    y_pred = df_combined[PREDICTED_CLASS]
    accuracy = sum(y_test == y_pred) / len(y_test)
    print('Accuracy for the combined model:', accuracy)

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion_m = confusion_matrix(y_test, y_pred)
    # use heatmap to plot confusion matrix
    # if PLOT_ENABLED:
    sns.heatmap(confusion_m, annot=True)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Combined Model')
    plt.show()

if __name__ == '__main__':
    random_forest_biax_ten = classify_biax_tension()
    random_forest_planar_comp = classify_planar_compression()
    random_forest_planar_ten = classify_planar_tension()
    random_forest_simple_shear = classify_simple_shear()
    random_forest_uniax_comp = classify_uniax_compression()
    random_forest_uniax_ten = classify_uniax_tension()

    df_biax_tension_test_results = get_biax_tension_test_predictions(random_forest_biax_ten)
    df_planar_compression_test_results = get_planar_compression_test_predictions(random_forest_planar_comp)
    df_planar_tension_test_results = get_planar_tension_test_predictions(random_forest_planar_ten)
    df_simple_shear_test_results = get_simple_shear_test_predictions(random_forest_simple_shear)
    df_uniax_compression_test_results = get_uniax_compression_test_predictions(random_forest_uniax_comp)
    df_uniax_tension_test_results = get_uniax_tension_test_predictions(random_forest_uniax_ten)

    # merge the results on sample name
    df_combined = pd.merge(df_biax_tension_test_results, df_planar_compression_test_results, on=feature_extractor.SAMPLE_NAME, how='outer')
    df_combined = pd.merge(df_combined, df_planar_tension_test_results, on=feature_extractor.SAMPLE_NAME, how='outer')
    df_combined = pd.merge(df_combined, df_simple_shear_test_results, on=feature_extractor.SAMPLE_NAME, how='outer')
    df_combined = pd.merge(df_combined, df_uniax_compression_test_results, on=feature_extractor.SAMPLE_NAME, how='outer')
    df_combined = pd.merge(df_combined, df_uniax_tension_test_results, on=feature_extractor.SAMPLE_NAME, how='outer')

    summarize_votes(df_combined)
    evaluate_combined_model(df_combined)
    print('done')


    





