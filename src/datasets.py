import csv

label_to_index_map= {"N":0, "D":1, "G":2, "C":3, "A":4, "H":5, "M":6, "O":7}

def label_to_index(label):

    return label_to_index_map.get(label, -1)

def read_csv(file_path, has_header=True):
    image_path = []
    label=[]
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_path.append(row[0])
            label.append(row[1])
        return image_path, label

if __name__== "__main__":  
    image_path, label= read_csv('data\processed_train_ODIR-5K.csv') 
    for path, label in zip(image_path, label):
        print(f"Image Path: {path}, Label: {label}")
