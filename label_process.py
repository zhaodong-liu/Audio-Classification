import numpy as np

def save_labels_to_npz(txt_file_path, npz_file_path):
    # read line by line
    with open(txt_file_path, 'r') as file:
        labels = file.readlines()
    
    # convert the data into integers and store them in a NumPy array
    labels = np.array([int(label.strip()) for label in labels])

    print(labels.shape)
    print(labels)

    
    # save as .npz file
    np.savez(npz_file_path, labels=labels)


txt_file_path = 'train_label.txt'
npz_file_path = 'train_label.npz'
save_labels_to_npz(txt_file_path, npz_file_path)
