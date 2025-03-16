import jax.numpy as jnp
import jax_dataloader as jdl

class Feeder_snr(jdl.Dataset):
    """ Feeder for label inputs """
    def __init__(self, data_path, label_path):
        self._data = jnp.load(data_path)
        self._label = jnp.load(label_path)
    
    def __getitem__(self, idx):
    # expand axis(-1)
        return {'data': self._data[idx, ..., None], 'label': self._label[idx,0].astype(jnp.int32)}
    def __len__(self):
        return len(self._data)

class Feeder_device(jdl.Dataset):
    """ Feeder for label inputs """
    def __init__(self, data_path, label_path, snr):
        self._data = jnp.load(data_path)
        self._data = self._data.reshape(3,-1,data_path.shape[1],data_path.shape[2])
        self._label = jnp.load(label_path).reshape(3,-1)

        for i in range(3):
            if snr == self._label[i][0][0]:
                self._label = self._label[i]
                self._data = self._data[i]
                break
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # get data
        return {'data': self._data[idx, ..., None], 'label': self._label[idx].astype(jnp.int32)}

class Feeder_label(jdl.Dataset):
    """ Feeder for label inputs """
    def __init__(self, data_path, label_path, snr):
        self._data = jnp.load(data_path)
        self._data = self._data.reshape(3,-1,data_path.shape[1],data_path.shape[2])
        self._label = jnp.load(label_path).reshape(3,-1)

        for i in range(3):
            if snr == self._label[i][0][0]:
                self._label = self._label[i]
                self._data = self._data[i]
                break
        
    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # get data
        return {'data': self._data[idx, ..., None], 'label': self._label[idx].astype(jnp.int32)}