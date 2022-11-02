import pickle
"""
keys of results={}:
test_loss
test_obj_loss
test_k_selection_loss
test_k_continuity_loss
test_accuracy
test_precision
test_recall
test_f1
test_confusion_matrix
test_mse
losses
preds
golds
rationales
"""

results_path = r"D:\text_nn\logs\t2_test.results"
f = open(results_path,'rb')
results = pickle.load(f)
f.close()

test_results = results["test_stats"]

avg_length = 0
n_test = len(test_results["rationales"])

for i in range(0, n_test):
    l = 0
    rationale = test_results["rationales"][i]
    text = results["test_data"][i]["text"]
    print(text)
    print(rationale)
    #print()
    rationale = test_results["rationales"][i]
    
    break
    r = rationale.split(" ")
    for token in r:
        if token != "_":
            l += 1
    avg_length += l
    
    #print("gold:", test_results["golds"][i])
    #print("pred:", test_results["preds"][i])
    #print("=========================================")

print("avg_length:", avg_length/n_test)
print("test_accuracy:", test_results["test_accuracy"])
print("selection_loss:", test_results["test_k_selection_loss"])
print("continuity_loss:", test_results["test_k_continuity_loss"])











