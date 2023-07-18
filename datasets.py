class TotalSegmenter_Dataset():
    def __init__(self, path, transform):
        super().__init__()
        
        self.path = path
        self.transform = transform

        self.csv = pd.read_csv(pd.read_csv(self.path + 'meta.csv', delimiter=';'))

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        pass