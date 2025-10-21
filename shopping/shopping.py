import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open("shopping.csv") as f:
        reader = csv.reader(f)
        next(reader)
        evidence = []
        labels = []
        for row in reader:
            row_list = []
            row_list.append(int(row[0]))
            row_list.append(float(row[1]))
            row_list.append(int(row[2]))
            row_list.append(float(row[3]))
            row_list.append(int(row[4]))
            row_list.append(float(row[5]))
            row_list.append(float(row[6]))
            row_list.append(float(row[7]))
            row_list.append(float(row[8]))
            row_list.append(float(row[9]))
            if row[10] == "Jan":
                row_list.append(0)
            elif row[10] == "Feb":
                row_list.append(1)
            elif row[10] == "Mar":
                row_list.append(2)
            elif row[10] == "Apr":
                row_list.append(3)
            elif row[10] == "May":
                row_list.append(4)
            elif row[10] == "June":
                row_list.append(5)
            elif row[10] == "Jul":
                row_list.append(6)
            elif row[10] == "Aug":
                row_list.append(7)
            elif row[10] == "Sep":
                row_list.append(8)
            elif row[10] == "Oct":
                row_list.append(9)
            elif row[10] == "Nov":
                row_list.append(10)
            elif row[10] == "Dec":
                row_list.append(11)
            row_list.append(int(row[11]))
            row_list.append(int(row[12]))
            row_list.append(int(row[13]))
            row_list.append(int(row[14]))
            row_list.append(1) if row[15] == "Returning_Visitor" else row_list.append(0)
            row_list.append(1) if row[16] == "TRUE" else row_list.append(0)
            labels.append(1) if row[17] == "TRUE" else labels.append(0)
            evidence.append(row_list)
    return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    pos_act = 0
    neg_act = 0
    pos_pred = 0
    neg_pred = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            neg_act += 1
            if predictions[i] == 0:
                neg_pred += 1
        else:
            pos_act += 1
            if predictions[i] == 1:
                pos_pred += 1
    sensitivity = pos_pred / pos_act
    specificity = neg_pred / neg_act
    return sensitivity, specificity


if __name__ == "__main__":
    main()
