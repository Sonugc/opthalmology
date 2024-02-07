from torch.utils.data import DataLoader
from src.datasets import ODIRDataset
from src.transforms import transforms


Batch_size = 10
train_csv_path = r"data/processed_train_ODIR-5K.csv"
val_csv_path= r"data/processed_val_ODIR-5K.csv"
test_csv_path= r"data/processed_test_ODIR-5k.csv"

train_dataset= ODIRDataset(csv_path=train_csv_path ,transforms=transforms,has_labels=True)
val_dataset=ODIRDataset(csv_path=val_csv_path, transforms=transforms,has_labels=True)
test_dataset=ODIRDataset(csv_path=test_csv_path, transforms=transforms,has_labels=False)

train_dataloader=  DataLoader(train_dataset, batch_size= Batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=Batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=Batch_size)

if __name__ == "__main__":
    images, labels = next(iter(train_dataloader))

    print(images, labels)