from torch.utils.data import Dataset, DataLoader

class aligned_celebA(Dataset):
    '''
        Aligned CelebA dataset
        Avaliable from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
    '''

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.csv.iloc[idx, 0])
        image = io.imread(img_name)
        image = cv2.resize(image, (128,128), interpolation=cv2.INTER_CUBIC) #TODO: Review image resizing technique
        image = np.rollaxis(image, 2, 0)
        
        #Normalise
        image = image/256
        
        return image