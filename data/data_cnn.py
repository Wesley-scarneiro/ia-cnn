class DataCnn:

    def __init__(self, 
                 train_images, train_labels, 
                 val_images, val_labels, 
                 test_images, test_labels) -> None:
        self.train_images = train_images
        self.train_labels = train_labels
        self.val_images = val_images
        self.val_labels = val_labels
        self.test_images = test_images
        self.test_labels = test_labels