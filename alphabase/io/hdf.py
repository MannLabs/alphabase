import h5py
import numpy as np
import pandas as pd
import re
import contextlib
import time


class HDF_Object(object):
    '''
    A generic class to access HDF components.
    '''

    @property
    def read_only(self):
        return self._read_only

    @property
    def truncate(self):
        return self._truncate

    @property
    def file_name(self):
        return self._file_name

    @property
    def name(self):
        return self._name

    def __eq__(self, other):
        return (
            self.file_name == other.self.file_name
        ) and (
            self.name == other.name
        )

    @contextlib.contextmanager
    def editing_mode(
        self,
        read_only: bool = False,
        truncate: bool = True
    ):
        original_read_only = self.read_only
        original_truncate = self.truncate
        try:
            self.set_read_only(read_only)
            self.set_truncate(truncate)
            yield self
        finally:
            self.set_read_only(original_read_only)
            self.set_truncate(original_truncate)

    @property
    def metadata(self):
        with h5py.File(self.file_name, "a") as hdf_file:
            return dict(hdf_file[self.name].attrs)

    def __init__(
        self,
        *,
        file_name: str,
        name: str,
        read_only: bool = True,
        truncate: bool = False,
    ):
        object.__setattr__(self, "_read_only", read_only)
        object.__setattr__(self, "_truncate", truncate)
        object.__setattr__(self, "_file_name", file_name)
        object.__setattr__(self, "_name", name)
        for key, value in self.metadata.items():
            object.__setattr__(self, key, value)

    def set_read_only(self, read_only: bool = True):
        object.__setattr__(self, "_read_only", read_only)

    def set_truncate(self, truncate: bool = True):
        object.__setattr__(self, "_truncate", truncate)

    def __setattr__(self, name, value):
        if self.read_only:
            raise AttributeError("Cannot set read-only attributes")
        elif not isinstance(name, str):
            raise KeyError(f"Attribute name '{name}' is not a string")
        elif not bool(re.match(r'^[a-zA-Z_][\w.-]*$', name)):
            raise KeyError(f"Invalid attribute name: {name}")
        if (not self.truncate) and (name in self.metadata):
            raise KeyError(
                f"Attribute '{name}' cannot be truncated"
            )
        if isinstance(value, (str, bool, int, float)):
            with h5py.File(self.file_name, "a") as hdf_file:
                hdf_object = hdf_file[self.name]
                hdf_object.attrs[name] = value
                object.__setattr__(self, name, value)
        else:
            raise NotImplementedError(
                f"Type '{type(name)}' is invalid for attribute {name}. "
                "Only (str, bool, int, float) types are accepted."
            )


class HDF_Group(HDF_Object):

    def __init__(
        self,
        *,
        file_name: str,
        name: str,
        read_only: bool = True,
        truncate: bool = False,
    ):
        super().__init__(
            file_name=file_name,
            name=name,
            read_only=read_only,
            truncate=truncate,
        )
        for dataset_name in self.dataset_names:
            dataset = HDF_Dataset(
                file_name=self.file_name,
                name=f"{self.name}/{dataset_name}",
                read_only=self.read_only,
                truncate=self.truncate,
            )
            object.__setattr__(self, dataset_name, dataset)
        for group_name in self.group_names:
            group = HDF_Group(
                file_name=self.file_name,
                name=f"{self.name}/{group_name}",
                read_only=self.read_only,
                truncate=self.truncate,
            )
            object.__setattr__(self, group_name, group)
        for dataframe_name in self.dataframe_names:
            dataframe = HDF_Dataframe(
                file_name=self.file_name,
                name=f"{self.name}/{dataframe_name}",
                read_only=self.read_only,
                truncate=self.truncate,
            )
            object.__setattr__(self, dataframe_name, dataframe)

    def __len__(self):
        return sum([len(component) for component in self.components])

    @property
    def group_names(self):
        return self.components[0]

    @property
    def dataset_names(self):
        return self.components[1]

    @property
    def dataframe_names(self):
        return self.components[2]

    @property
    def groups(self):
        return [
            self.__getattribute__(name) for name in self.group_names
        ]

    @property
    def datasets(self):
        return [
            self.__getattribute__(name) for name in self.dataset_names
        ]

    @property
    def dataframes(self):
        return [
            self.__getattribute__(name) for name in self.dataframe_names
        ]

    @property
    def components(self):
        group_names = []
        dataset_names = []
        datafame_names = []
        with h5py.File(self.file_name, "a") as hdf_file:
            hdf_object = hdf_file[self.name]
            for name in sorted(hdf_object):
                if isinstance(hdf_object[name], h5py.Dataset):
                    dataset_names.append(name)
                else:
                    if "is_pd_dataframe" in hdf_object[name].attrs:
                        if hdf_object[name].attrs["is_pd_dataframe"]:
                            datafame_names.append(name)
                    else:
                        group_names.append(name)
        return group_names, dataset_names, datafame_names

    def set_read_only(self, read_only: bool = True):
        super().__setattr__(self, "_read_only", read_only)
        for dataset_name in self.dataset_names:
            self.__getattribute__(dataset_name).set_read_only(read_only)
        for group_name in self.group_names:
            self.__getattribute__(group_name).set_read_only(read_only)
        for dataframe_name in self.dataframe_names:
            self.__getattribute__(dataframe_name).set_read_only(read_only)

    def set_truncate(self, truncate: bool = True):
        super().__setattr__(self, "_truncate", truncate)
        for dataset_name in self.dataset_names:
            self.__getattribute__(dataset_name).set_truncate(truncate)
        for group_name in self.group_names:
            self.__getattribute__(group_name).set_truncate(truncate)
        for dataframe_name in self.dataframe_names:
            self.__getattribute__(dataframe_name).set_truncate(truncate)

    def __setattr__(self, name, value):
        try:
            super().__setattr__(name, value)
        except NotImplementedError:
            if not self.truncate:
                if name in self.group_names:
                    raise KeyError(
                        f"Group name '{name}' cannot be truncated"
                    )
                elif name in self.dataset_names:
                    raise KeyError(
                        f"Dataset name '{name}' cannot be truncated"
                    )
                elif name in self.dataframe_names:
                    raise KeyError(
                        f"Dataframe name '{name}' cannot be truncated"
                    )
            if isinstance(value, (np.ndarray, pd.core.series.Series)):
                self.add_dataset(name, value)
            elif isinstance(value, (dict, pd.DataFrame)):
                self.add_group(name, value)
            else:
                raise NotImplementedError(
                    f"Type '{type(value)}' is invalid for attribute {name}",
                    "Only (str, bool, int, float, np.ndarray, "
                    "pd.core.series.Series, dict pd.DataFrame) types are "
                    "accepted."
                )

    def add_dataset(
        self,
        name: str,
        array: np.ndarray,
    ):
        with h5py.File(self.file_name, "a") as hdf_file:
            hdf_object = hdf_file[self.name]
            if name in hdf_object:
                del hdf_object[name]
            if isinstance(array, (pd.core.series.Series)):
                array = array.values
            # if array.dtype == np.dtype('O'):
            #     print("YAR")
            #     # dtype = h5py.string_dtype(encoding='utf-8')
            #     dtype = h5py.vlen_dtype(str)
            # else:
            #     dtype = array.dtype
            #     # data=value_.astype(str).values,
            #     # # dtype=h5py.string_dtype(encoding='utf-8')
            #     # dtype=h5py.vlen_dtype(str),
            try:
                hdf_object.create_dataset(
                    name,
                    data=array,
        #             TODO
                    compression="lzf",
        #             # compression="gzip" if compress else None, # TODO slower to make, faster to load?
                    shuffle=True,
                    chunks=True,
                    # chunks=array.shape,
                    # dtype=dtype,
                    maxshape=tuple([None for i in array.shape]),
                )
            except TypeError:
                raise NotImplementedError(
                    f"Type {array.dtype} is not understood. "
                    "If this is a string format, try to cast it to "
                    "np.dtype('O') as possible solution."
                )
            dataset = HDF_Dataset(
                file_name=self.file_name,
                name=f"{self.name}/{name}",
                read_only=self.read_only,
                truncate=self.truncate,
            )
            dataset.last_updated = time.asctime()
            # object.__setattr__(self, "_shape", self._shape + 1) # TODO move to Dataframe?
            object.__setattr__(self, name, dataset)

    def add_group(
        self,
        name: str,
        group: dict,
    ):
        with h5py.File(self.file_name, "a") as hdf_file:
            hdf_object = hdf_file[self.name]
            if name in hdf_object:
                del hdf_object[name]
            hdf_object.create_group(name)
        if isinstance(group, pd.DataFrame):
            group = dict(group)
            group["is_pd_dataframe"] = True
            new_group = HDF_Dataframe(
                file_name=self.file_name,
                name=f"{self.name}/{name}",
                read_only=self.read_only,
                truncate=self.truncate,
            )
        else:
            new_group = HDF_Group(
                file_name=self.file_name,
                name=f"{self.name}/{name}",
                read_only=self.read_only,
                truncate=self.truncate,
            )
        for key, value in group.items():
            new_group.__setattr__(key, value)
        new_group.last_updated = time.asctime()
        object.__setattr__(self, name, new_group)


class HDF_Dataset(HDF_Object):

    def __len__(self):
        return self.shape[0]

    @property
    def dtype(self):
        with h5py.File(self.file_name, "a") as hdf_file:
            return hdf_file[self.name].dtype

    @property
    def shape(self):
        with h5py.File(self.file_name, "a") as hdf_file:
            return hdf_file[self.name].shape

    @property
    def values(self):
        return self[...]

    def __getitem__(self, keys):
        with h5py.File(self.file_name, "a") as hdf_file:
            hdf_object = hdf_file[self.name]
            if h5py.check_string_dtype(hdf_object.dtype) is not None:
                hdf_object = hdf_object.asstr()
            return hdf_object[keys]

    def append(self, data):
        with h5py.File(self.file_name, "a") as hdf_file:
            hdf_object = hdf_file[self.name]
            new_shape = tuple(
                [i + j for i, j in zip(self.shape, data.shape)]
            )
            old_size = len(self)
            hdf_object.resize(new_shape)
            hdf_object[old_size:] = data

    def set_slice(self, selection, values):
        with h5py.File(self.file_name, "a") as hdf_file:
            hdf_object = hdf_file[self.name]
            hdf_object[selection] = values


class HDF_Dataframe(HDF_Group):

    @property
    def dtype(self):
        dtypes = []
        for column_name in self.dataset_names:
            dtype = self.__getattribute__(column_name).dtype
            dtypes.append(dtype)
        return list(dtypes)

    @property
    def columns(self):
        return self.dataset_names

    def __len__(self):
        return len(self.__getattribute__(self.dataset_names[0]))

    @property
    def values(self):
        return self[...]

    def __getitem__(self, keys):
        df_dict = {}
        for column_name in self.dataset_names:
            dataset = self.__getattribute__(column_name)
            if isinstance(dataset, HDF_Dataset):
                df_dict[column_name] = dataset[keys]
        return pd.DataFrame(df_dict)

    def append(self, data):
        for column_name in self.dataset_names:
            dataset = self.__getattribute__(column_name)
            if isinstance(dataset, HDF_Dataset):
                dataset.append(data[column_name])

    def set_slice(self, selection, values):
        for column_name in self.dataset_names:
            dataset = self.__getattribute__(column_name)
            dataset.set_slice(selection, values[column_name])


class HDF_File(HDF_Group):

    def __init__(
        self,
        file_name: str,
        *,
        read_only: bool = True,
        truncate: bool = False,
        delete_existing: bool = False,
    ):
        if delete_existing:
            mode = "w"
        else:
            mode = "a"
        with h5py.File(file_name, mode):#, swmr=True):
            pass
        super().__init__(
            file_name=file_name,
            name="/",
            read_only=read_only,
            truncate=truncate,
        )
