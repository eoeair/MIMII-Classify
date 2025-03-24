import numpy as np
import grain.python as grain

# Note: Inheriting `grain.RandomAccessDataSource` is optional but recommended.
class Feeder_snr(grain.RandomAccessDataSource):
    """ Feeder for snr inputs """
    def __init__(self, data_path, label_path):
        self._data = np.load(data_path)
        self._label = np.load(label_path)
    
    def __getitem__(self, idx):
    # expand axis(-1)
        return {'data': self._data[idx, ..., None], 'label': self._label[idx,0].astype(np.int32)}
    def __len__(self):
        return len(self._data)

class Feeder_device(grain.RandomAccessDataSource):
    """ Feeder for device inputs """
    def __init__(self, data_path, label_path, snr):
        self._data = np.load(data_path)
        self._data = self._data.reshape(3,-1,self._data.shape[1],self._data.shape[2])
        self._label = np.load(label_path).reshape(3,-1)

        for i in range(3):
            if snr == self._label[i][0][0]:
                self._label = self._label[i]
                self._data = self._data[i]
                break
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # get data
        return {'data': self._data[idx, ..., None], 'label': self._label[idx].astype(np.int32)}

class Feeder_label(grain.RandomAccessDataSource):
    """ Feeder for label inputs """
    def __init__(self, data_path, label_path, snr):
        self._data = np.load(data_path)
        self._data = self._data.reshape(3,-1,self._data.shape[1],self._data.shape[2])
        self._label = np.load(label_path).reshape(3,-1)

        for i in range(3):
            if snr == self._label[i][0][0]:
                self._label = self._label[i]
                self._data = self._data[i]
                break
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # get data
        return {'data': self._data[idx, ..., None], 'label': self._label[idx].astype(np.int32)}

# modal: snr, device, label
def load_data(modal, snr=0, num_workers=8, num_epochs=10, batch_size=256):
    if modal == 'snr':
        train_source = Feeder_snr('data/train_data.npy','data/train_label.npy')
        test_source = Feeder_snr('data/test_data.npy','data/test_label.npy')
    elif modal == 'device':
        train_source = Feeder_device('data/train_data.npy','data/train_label.npy', snr)
        test_source = Feeder_device('data/test_data.npy','data/test_label.npy', snr)
    elif modal == 'label':
        train_source = Feeder_label('data/train_data.npy','data/train_label.npy', snr)
        test_source = Feeder_label('data/test_data.npy','data/test_label.npy', snr)
    
    train_sampler = grain.IndexSampler(
        num_records=len(train_source),
        num_epochs=num_epochs,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=0)
    test_sampler = grain.IndexSampler(
        num_records=len(test_source),
        num_epochs=1,
        shard_options=grain.NoSharding(),
        shuffle=True,
        seed=0)

    train_loader = grain.DataLoader(
        data_source=train_source,
        operations=[grain.Batch(batch_size=batch_size)],
        sampler=train_sampler,
        worker_count=num_workers
        )
    test_loader = grain.DataLoader(
        data_source=test_source,
        operations=[grain.Batch(batch_size=batch_size)],
        sampler=test_sampler,
        worker_count=num_workers
        )
    
    return train_loader, test_loader