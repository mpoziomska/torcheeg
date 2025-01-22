from typing import Any, Callable, Dict, Tuple, Union
import pandas as pd
import mne
import numpy as np

from .base_dataset import BaseDataset
from ...utils import get_random_dir_path
import logging
import gc

from eegunirep.utils.electrode_utils import CHAN_LIST, CHNAMES_MAPPING, apply_mor_data_hack_fix

import warnings
from scipy.signal import BadCoefficients

log = logging.getLogger('torcheeg')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

class CSVFolderDataset(BaseDataset):
    def __init__(self,
                 config,
                 csv_path: str = './data.csv',
                 online_transform: Union[None, Callable] = None,
                 label_transform: Union[None, Callable] = None,
                 io_path: Union[None, str] = None,
                 io_size: int = (1048576 * (2**6)),
                 io_mode: str = 'lmdb',
                 num_worker: int = 0,
                 verbose: bool = True,
                 **kwargs):
        self.config = config
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix='datasets')

        params = {
            'csv_path': csv_path,
            'read_fn': None,
            'online_transform': online_transform,
            'offline_transform': None,
            'label_transform': label_transform,
            'io_path': io_path,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose
        }

        params.update(kwargs)
        super().__init__(**params)
        # save all arguments to __dict__
        self.__dict__.update(params)

    def process_record(self, file: Any = None,
                       offline_transform: Union[None, Callable] = None,
                       read_fn: Union[None, Callable] = None,
                       **kwargs):

        trial_info = file
        file_path = trial_info['file_path']

        # log.info(f"FILE PATH {file_path}")
        edf = mne.io.read_raw_edf(input_fname=file_path, preload=True)
        edf = apply_mor_data_hack_fix(edf=edf, edf_path=file_path, institution_id=trial_info['institution_id'])
        with mne.utils.use_log_level("error"):
            # ujednolica nazwy elektrod oraz ich kolejność,
            # jeśli nie da rady, bo np. brak wystarczającej ilości
            # elektrod, to rzuca wyjątkiem
            i = 0
            while True:
                try:
                    edf = edf.rename_channels(CHNAMES_MAPPING[i])
                    edf = edf.reorder_channels(CHAN_LIST)
                    break
                except ValueError:
                    i += 1

                if i == len(CHNAMES_MAPPING):
                    raise ValueError(
                        f"channels rename/reordering error, available channels\n{edf.info['ch_names']}"
                        f", required channels: {CHAN_LIST}"
                    )

        edf: mne.io.Raw = edf.set_montage("standard_1020", match_case=False) # Czy to jest potrzebne?
        Fs = edf.info['sfreq']
        new_Fs = self.config['preprocess']['new_Fs']

        f_stop = self.config['preprocess']['f_stop']
        f_pass = self.config['preprocess']['f_pass']
        
        iir_params = dict(ftype='butter', output='sos', gpass=1, gstop=20)
        
        iir_params = mne.filter.construct_iir_filter(iir_params,
                                                    f_pass=f_pass,
                                                    f_stop=f_stop,
                                                    sfreq=Fs,
                                                    btype='lowpass',
                                                    return_copy=False)

        edf = edf.filter(iir_params=iir_params, l_freq=0, h_freq=0, method='iir')
        edf = edf.resample(sfreq=new_Fs)

        eeg_raw_signal = edf.get_data(picks=CHAN_LIST, units='uV', tmin=self.config['preprocess']['tmin_s'], tmax=self.config['preprocess']['tmax_s'])[:, None]
        eid = file_path.split('/')[-1]
        record_info = {
                **trial_info, 'Fs': new_Fs,
                'clip_id': eid
            }
        for i in range(1):
            yield {'eeg': eeg_raw_signal, 'key': eid, 'info': record_info}

    def set_records(self, csv_path: str = './data.csv', **kwargs):
        # read csv
        df_info = pd.read_csv(csv_path)
        assert 'file_path' in df_info.columns, 'file_path is required in csv file.'

        # df to a list of dict, each dict is a row
        df_list = df_info.to_dict('records')

        return df_list

    def __getitem__(self, index: int) -> Tuple[any, any, int, int, int]:
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg_record = str(info['_record_id'])
        eeg = self.read_eeg(eeg_record, eeg_index).reshape(len(CHAN_LIST), -1)

        signal = eeg
        label = info

        if self.online_transform:
            try:
                signal = self.online_transform(eeg)
            except TypeError:
                signal = self.online_transform(eeg=eeg)

        if self.label_transform:
            label = self.label_transform(y=info)['y']
        # del eeg
        # gc.collect()
        return signal, label, info

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'csv_path': self.csv_path,
                'read_fn': self.read_fn,
                'online_transform': self.online_transform,
                'offline_transform': self.offline_transform,
                'label_transform': self.label_transform,
                'before_trial': self.before_trial,
                'after_trial': self.after_trial,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })