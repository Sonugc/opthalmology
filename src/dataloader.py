from torch.utils.data import DataLoader
from datasets import ODIRDataset
from transforms import transforms



train_csv_path = r"data\processed_train_ODIR-5K.csv"
val_csv_path= r"data\processed_val_ODIR-5K.csv"
test_csv_path= r"data\processed_test_ODIR-5k.csv"

train_dataset= ODIRDataset(csv_path=train_csv_path ,transforms=transforms,has_labels=True)
val_dataset=ODIRDataset(csv_path=val_csv_path, transforms=transforms,has_labels=True)
test_dataset=ODIRDataset(csv_path=test_csv_path, transforms=transforms,has_labels=False)

train_dataloader=  DataLoader(train_dataset, batch_size=10, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=10)
test_dataloader = DataLoader(test_dataset, batch_size=10)

if __name__ == "__main__":
    images, labels = next(iter(train_dataloader))

    print(images, labels)