import csv
import sys
import matplotlib.pyplot as plt
import numpy as np


score = []
hospitalaccountidlist = []
qsofalist = []
y = []


#working here - cha

with open('Redacted code', newline='') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        hospitalaccountid = int(row['hospitalaccountid'])
        sepsis = float(row['sepsis'])
        qsofa = float(row['qsofa'])
        modelscoreFull = float(row['full_modelscore'])

        score.append(modelscoreFull)
        y.append(sepsis)
        qsofalist.append(qsofa)
        hospitalaccountidlist.append(hospitalaccountid)

print(" y list is:", y)
print(" score list is" , score)
print(" hospitalaccountidlist is", hospitalaccountidlist)
sepsisprevalence = sum(y)/len(y)

score = np.array(score)
qsofalist = np.array(qsofalist)
y = np.array(y)


#############
# false positive rate
fpr = []
# true positive rate
tpr = []
ppr=[]
npr=[]


thresholds = np.sort(score)
# print("sorting score -thresholds",thresholds)
thresholds = np.insert(thresholds,0, np.min(thresholds)-0.1)
# print("thresholds after insert",thresholds)
# print("minimum thresholds",np.min(thresholds))


# get number of positive and negative examples in the dataset
P = sum(y)
N = len(y) - P

# iterate through all thresholds and determine fraction of true positives
# and false positives found at this threshold

for thresh in thresholds:
    FP=0
    TP=0
    for i in range(len(score)):
        if (score[i] > thresh):
            if y[i] == 1:
                TP = TP + 1
            if y[i] == 0:
                FP = FP + 1

    fpr.append(FP/float(N))
    tpr.append(TP/float(P))

optimal_idx = np.argmax(np.array(tpr) - np.array(fpr))
optimal_threshold = thresholds[optimal_idx]
print("optimal_threshold",optimal_threshold)
print("optimal_idx",optimal_idx)


FP=0
TP=0
TN=0
FN=0
optimal_threshold = 0.45
for i in range(len(score)):
    if (score[i] > optimal_threshold):
        if y[i] == 1:
            TP = TP + 1
        if y[i] == 0:
            FP = FP + 1
    if (score[i] < optimal_threshold):
        if y[i] == 0:
            TN = TN + 1
        if y[i] == 1:
            FN = FN + 1


print("sensitivity is ", TP/(TP+FN), " and the specificity is ", TN/(TN+FP), " at a optimal threshold of ", optimal_threshold, "with an accuracy of", (TP+TN)/(TP+TN+FP+FN), "for the Full SRI model")
print("sepsis count is ", P, " non-sepsis count is ", N, " total patient count is ", P+N, "PPV", TP/(TP+FP), "NPV", TN/(FN+TN))
# cha positive predictive rate 65% which is (TP/(TP+FP))*100
print("sensitivity is ", TP/(TP+FN))
print("the specificity is ", TN/(TN+FP))
print("PPV", TP/(TP+FP))
print("NPV", TN/(FN+TN))
print("accuracy of", (TP+TN)/(TP+TN+FP+FN))

#end by cha ppr
print("SRI tpr list")
print(tpr)
print("SRI fpr list")
print(fpr)

FP = 0
TP = 0
TN = 0
FN = 0

columns = ["threshold >=","sensitivity", "specificity", "PPV", "NPV", "accuracy", "F1"]
out_file = open(
        "cha_MGH_perf_characteristics.csv",
        "w")
writer = csv.DictWriter(out_file, fieldnames=columns)
writer.writeheader()
thresholdvalues = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
for value in thresholdvalues:
    FP=0
    TP=0
    TN=0
    FN=0
    for i in range(len(score)):
        if (score[i] > value):
            if y[i] == 1:
                TP = TP + 1
            if y[i] == 0:
                FP = FP + 1
        if (score[i] < value):
            if y[i] == 0:
                TN = TN + 1
            if y[i] == 1:
                FN = FN + 1
    precision = TP / ( TP + FP )
    recall = TP / (TP + FN )

    writer.writerow({
        "threshold >=": value,
        "sensitivity": TP/(TP+FN),
        "specificity": TN/(TN+FP),
        "PPV": TP/(TP+FP),
        "NPV": TN/(FN+TN),
        "accuracy": (TP+TN)/(TP+TN+FP+FN),
	"F1": (2*precision*recall)/(precision+recall)
    })

