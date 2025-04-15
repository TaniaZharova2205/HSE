import numpy as np

class DataLoader(object):
    """
    Tool for shuffling data and forming mini-batches
    """
    def __init__(self, X, y, batch_size=1, shuffle=False):
        """
        :param X: dataset features
        :param y: dataset targets
        :param batch_size: size of mini-batch to form
        :param shuffle: whether to shuffle dataset
        """
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_id = 0  # use in __next__, reset in __iter__

    # использовала ChatGPT с моделью GPT-4-turbo. Промт:
    '''
        Почему не работает функция __len__(self) DataLoader:
        def __len__(self) -> int:
            return self.X.shape[0] // self.batch_size
    '''
    def __len__(self) -> int:
        return int(np.ceil(self.X.shape[0] / self.batch_size)) 

    def num_samples(self) -> int:
        """
        :return: number of data samples
        """
        return self.X.shape[0]

    def __iter__(self):
        """
        Shuffle data samples if required
        :return: self
        """
        self.batch_id = 0
        if self.shuffle:
            indices = np.random.permutation(self.X.shape[0]) # перемешиваем индексы
            self.X = self.X[indices]
            self.y = self.y[indices]
        return self

    # алгоритм взят из https://stackoverflow.com/questions/19151/how-to-build-a-basic-iterator 
    def __next__(self):
        if self.batch_id < len(self):    
            start = self.batch_id * self.batch_size # индекс первого элемента в батче
            end = min(start + self.batch_size, self.X.shape[0]) # индекс последнего элемента в батче
            X_batch = self.X[start:end]
            y_batch = self.y[start:end]
            
            self.batch_id += 1 # увеличиваем номер батча на 1, чтобы получить следующий батч в следующем вызове
            return X_batch, y_batch
        raise StopIteration
