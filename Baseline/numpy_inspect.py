import numpy as np

def inspect_and_compare(npz_file_path1, npz_file_path2):
    with np.load(npz_file_path1) as data1, np.load(npz_file_path2) as data2:
        print(f"Inspecting '{npz_file_path1}'")
        for key in data1.files:
            print(f"Key: {key}, Shape of array: {data1[key].shape}")

        print(f"Inspecting '{npz_file_path2}'")
        for key in data2.files:
            print(f"Key: {key}, Shape of array: {data2[key].shape}")

        # Compare X_train and X_test
        X_train1 = data1['X_train']
        X_train2 = data2['X_train']
        X_test1 = data1['X_test']
        X_test2 = data2['X_test']

        print("Sample data comparison, X_train1 vs X_train2 first 10 samples:", np.array_equal(X_train1[:10], X_train2[:10]))
        print("Sample data comparison, X_test1 vs X_test2 first 10 samples:", np.array_equal(X_test1[:10], X_test2[:10]))

        # Compare y_train and y_test
        y_train1 = data1['y_train']
        y_train2 = data2['y_train']
        y_test1 = data1['y_test']
        y_test2 = data2['y_test']

        # Print unique labels and their counts
        unique_labels1, counts1 = np.unique(y_train1, return_counts=True)
        unique_labels2, counts2 = np.unique(y_train2, return_counts=True)
        unique_test_labels1, test_counts1 = np.unique(y_test1, return_counts=True)
        unique_test_labels2, test_counts2 = np.unique(y_test2, return_counts=True)

        print("Unique labels and counts in y_train1:", dict(zip(unique_labels1, counts1)))
        print("Unique labels and counts in y_train2:", dict(zip(unique_labels2, counts2)))
        print("Unique labels and counts in y_test1:", dict(zip(unique_test_labels1, test_counts1)))
        print("Unique labels and counts in y_test2:", dict(zip(unique_test_labels2, test_counts2)))

        print("Sample data comparison, y_train1 vs y_train2 first 10 labels:", np.array_equal(y_train1[:10], y_train2[:10]))
        print("Sample data comparison, y_test1 vs y_test2 first 10 labels:", np.array_equal(y_test1[:10], y_test2[:10]))

        print("Mean and std of y_train1:", np.mean(y_train1), np.std(y_train1))
        print("Mean and std of y_train2:", np.mean(y_train2), np.std(y_train2))


# Usage eXample
file1 = 'data/Windows_PE/real_world/malware.npz'
file2 = 'data/Windows_PE/real_world/malware_true.npz'
inspect_and_compare(file1, file2)

file3 = 'data/Windows_PE/synthetic/malware.npz'
file4 = 'data/Windows_PE/synthetic/malware_true.npz'
inspect_and_compare(file3, file4)
