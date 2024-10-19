import json
import numpy as np
import pandas as pd

from pandas.api.types import (
    is_numeric_dtype,
    is_datetime64_any_dtype,
    is_categorical_dtype,
)


"""
Utilities to serialize and deserialize Pandas data objects.
"""


# ===========================================================================
# Data Frames
# ===========================================================================

class PdDataFrameJsonEncoder(json.JSONEncoder):
    """
    JSON encoder class for pandas DataFrame objects.
    """

    def default(self, obj):
        columns = {}
        series_encoder = PdSeriesJsonEncoder()
        series_encoder.omit_index()

        for vname in obj.columns:
            columns[vname] = series_encoder.default(obj[vname])

        index_encoder = PdIndexJsonEncoder()
        payload = {
            "column_values": columns,
            "row_index": index_encoder.default(obj.index),
            "column_names": index_encoder.default(obj.columns),
        }
        return payload


class PdDataFrameJsonDecoder(json.JSONDecoder):
    """
    JSON decoder class for pandas DataFrame objects.
    """
    def decode(self, json_string):
        payload = json.loads(json_string)
        return self.decode_from_dict(payload)

    def decode_from_dict(self, payload):
        series_decoder = PdSeriesJsonDecoder()
        index_decoder = PdIndexJsonDecoder()

        column_names = index_decoder.decode_from_dict(
            payload["column_names"])
        ans = pd.DataFrame(columns=column_names)

        for vname in column_names:
            # If the column name is an integer, then ans.loc[] is needed to
            # unambiguously refer to the correct column.  The payload may or
            # may not convert the integer to a string.
            try:
                ans.loc[:, vname] = series_decoder.decode_from_dict(
                    payload["column_values"][vname])
            except KeyError:
                if isinstance(vname, bool):
                    value = str(vname).lower()
                else:
                    value = str(vname)
                ans.loc[:, vname] = series_decoder.decode_from_dict(
                    payload["column_values"][value])

        # The names are set and are in the right order, but if the original
        # 'columns' attribute had other attributes set, they need to be
        # restored.  Reassigning 'columns' using the deserialized object does
        # that.
        ans.columns = column_names

        ans.index = index_decoder.decode_from_dict(payload["row_index"])
        return ans


# ===========================================================================
# Series
# ===========================================================================

class PdSeriesJsonEncoder(json.JSONEncoder):
    """
    JSON encode a Pandas Series object.
    """

    def __init__(self):
        self._index_encoder = PdIndexJsonEncoder()

    def omit_index(self):
        self._index_encoder = None

    def default(self, obj):
        dtype = str(obj.dtype)
        payload = {"dtype": dtype}

        if is_numeric_dtype(obj.dtype):
            payload["data"] = self.encode_numeric(obj)
        elif dtype == "category":
            payload["data"] = self.encode_categorical(obj)
        elif is_datetime64_any_dtype(dtype):
            payload["data"] = self.encode_datetime(obj)
        elif dtype == "object":
            payload["data"] = obj.tolist()

        if self._index_encoder is not None:
            payload["index"] = self._index_encoder.default(obj.index)

        return payload

    def encode_numeric(self, obj):
        values = obj.tolist()
        if obj.hasnans:
            na_locations = np.arange(len(obj))[obj.isna()]
            for i in na_locations:
                values[i] = "nan"
        return values

    def encode_categorical(self, obj):
        categories_encoder = PdIndexJsonEncoder()
        ans = {"categories": categories_encoder.default(obj.cat.categories),
               "values": obj.tolist()}
        return ans

    def encode_datetime(self, obj):
        return obj.astype(str).tolist()


class PdSeriesJsonDecoder(json.JSONDecoder):
    """
    Decode a JSON encoded Pandas Series object.
    """

    def decode(self, json_string):
        payload = json.loads(json_string)
        return self.decode_from_dict(payload)

    def decode_from_dict(self, payload):
        dtype = str(payload["dtype"])

        # Decoding supports a limited set of dtypes.  Add to this list as
        # needed.
        if is_numeric_dtype(dtype):
            series = self.decode_numeric(payload["data"], dtype)
        elif is_datetime64_any_dtype(dtype):
            series = self.decode_datetime(payload["data"], dtype)
        elif is_categorical_dtype(dtype):
            series = self.decode_categorical(payload["data"])
        elif dtype == "object":
            series = pd.Series(payload["data"], dtype=dtype)
        else:
            raise Exception(
                f"PdSeriesJsonDecoder encountered an unknown dtype: {dtype}.\n"
                "If desired, consider adding support for this type in the "
                "'decode_from_dict' method."
            )

        if "index" in payload.keys():
            index_decoder = PdIndexJsonDecoder()
            index = index_decoder.decode_from_dict(payload["index"])
            series.index = index

        return series

    def decode_numeric(self, values, dtype: str):
        return pd.Series(values, dtype=dtype)

    def decode_datetime(self, values, dtype):
        return pd.Series(values, dtype=dtype)

    def decode_object(self, values):
        return pd.Series(values, dtype="object")

    def decode_categorical(self, values):
        category_decoder = PdIndexJsonDecoder()
        categories = category_decoder.decode_from_dict(values["categories"])
        series = pd.Series(values["values"], dtype="category")
        series.cat.set_categories(categories)
        return series


# ===========================================================================
# Index
# ===========================================================================

class PdIndexJsonEncoder(json.JSONEncoder):
    """
    Pandas uses several types for the "Index" of its data frames and series
    objects.

    This code handles the special cases of
    * IntervalIndex
    * CategoricalIndex
    * DatetimeIndex

    and the generic type
    * Index

    JSON encoder for arbitrary pandas 'Index' types.  Each index type is
    explicitly handled as a special case.  When new types are encountered code
    for encoding them must be added manually.
    """

    def default(self, index):
        """
        :param obj:  The pandas Index object to be encoded.
        """
        payload = {}
        if isinstance(index, pd.IntervalIndex):
            payload["index_type"] = "IntervalIndex"
            payload["index_left"] = index.left.tolist()
            payload["index_right"] = index.right.tolist()
            payload["index_dtype"] = str(index.dtype)
            payload["index_closed"] = str(index.closed)
            payload["index_name"] = index.name

        elif isinstance(index, pd.CategoricalIndex):
            payload["index_type"] = "CategoricalIndex"
            payload = self.append_categorical_index(payload, index)

        elif isinstance(index, pd.DatetimeIndex):
            payload["index_type"] = "DatetimeIndex"
            payload["index_name"] = index.name
            payload["index_dtype"] = str(index.dtype)
            payload["values"] = [str(x) for x in index]

        elif isinstance(index, pd.Index):
            payload["index_type"] = "Index"
            payload["index_name"] = index.name
            if np.any([isinstance(x, pd.Interval) for x in index]):
                payload["contains_intervals"] = True
                payload["left"] = [x.left for x in index if isinstance(
                    x, pd.Interval)]
                payload["right"] = [x.right for x in index if isinstance(
                    x, pd.Interval)]
                payload["closed"] = [x.closed for x in index if isinstance(
                    x, pd.Interval)]
                payload["other_values"] = [x for x in index if not isinstance(
                    x, pd.Interval)]
            else:
                payload["contains_intervals"] = False
                payload["index_values"] = index.tolist()

            payload["index_dtype"] = str(index.dtype)

        else:
            raise Exception(
                f"Index of type {type(index)} could not be JSON encoded.")
        return payload

    def append_categorical_index(self, payload: dict, index):
        """Add entries to the payload describing the index.

        :param payload:
            The current JSON payload

        :return payload:
            The modified payload with the index appended.

        """

        index_dtype = str(index.dtype)
        payload["index_dtype"] = index_dtype
        payload["index_codes"] = index.codes.tolist()
        payload["index_ordered"] = index.ordered
        payload["index_name"] = index.name

        categories = index.categories
        if isinstance(categories, list):
            payload["category_storage_type"] = "list"
            payload["categories"] = categories
        elif isinstance(categories, pd.IntervalIndex):
            payload["category_storage_type"] = "IntervalIndex"
            payload["categories_left"] = categories.left.tolist()
            payload["categories_right"] = categories.right.tolist()
            payload["categories_dtype"] = str(categories.dtype)
            payload["categories_closed"] = categories.closed
            payload["categories_name"] = categories.name
        elif isinstance(categories, pd.Index):
            payload["category_storage_type"] = "Index"
            category_values = categories.values
            assert isinstance(category_values, np.ndarray)
            payload["categories"] = category_values.tolist()
            payload["categories_dtype"] = str(category_values.dtype)
            payload["categories_name"] = categories.name

        else:
            raise Exception(
                "I'm not sure how to JSON encode the categories from a "
                f"{type(categories)}, which are part of the index of "
                "this series."
            )

        return payload


class PdIndexJsonDecoder(json.JSONDecoder):
    def decode(self, json_string):
        payload = json.loads(json_string)
        return self.decode_from_dict(payload)

    def decode_from_dict(self, payload):
        index_type = payload["index_type"]
        if index_type == "IntervalIndex":
            left = payload["index_left"]
            right = payload["index_right"]
            index_dtype = payload["index_dtype"]
            closed = payload["index_closed"]
            name = payload["index_name"]
            index = pd.IntervalIndex.from_arrays(
                left, right, dtype=index_dtype, closed=closed, name=name
            )

        elif index_type == "CategoricalIndex":
            index = self.extract_categorical_index(payload)

        elif index_type == "DatetimeIndex":
            index = pd.DatetimeIndex(
                payload["values"], name=payload["index_name"],
                dtype=payload["index_dtype"]
            )

        elif index_type == "Index":
            name = payload["index_name"]
            index_dtype = payload["index_dtype"]
            if payload["contains_intervals"]:
                values = []
                for i in range(len(payload["left"])):
                    values.append(
                        pd.Interval(
                            payload["left"][i], payload["right"][i],
                            closed=payload["closed"][i]
                        )
                    )
                values += payload["other_values"]

            else:
                values = payload["index_values"]
            index = pd.Index(data=values, dtype=index_dtype, name=name)

        return index

    def extract_categorical_index(self, payload: dict):
        """Extract a CategoricalIndex from a JSON payload.

        :param payload:
            The current JSON payload

        :return index:
            A pd.CategoricalIndex extracted from the payload.

        """

        codes = payload["index_codes"]
        ordered = bool(payload["index_ordered"])
        index_dtype = payload["index_dtype"]
        name = payload["index_name"]
        category_storage_type = payload["category_storage_type"]
        if category_storage_type == "list":
            categories = pd.Series(payload["categories"])
        elif category_storage_type == "IntervalIndex":
            left = payload["categories_left"]
            right = payload["categories_right"]
            categories_dtype = payload["categories_dtype"]
            closed = payload["categories_closed"]
            categories_name = payload["categories_name"]
            categories = pd.IntervalIndex.from_arrays(
                left, right, dtype=categories_dtype, closed=closed,
                name=categories_name
            )
        elif category_storage_type == "Index":
            category_values = np.array(payload["categories"],
                                       dtype=payload["categories_dtype"])
            categories_name = payload["categories_name"]
            categories = pd.Index(data=category_values, name=categories_name)

        else:
            raise Exception(
                "I don't know how to deserialize a JSON string with "
                f"category_storage_type {category_storage_type}."
            )

        try:
            if False:  # categories.empty:
                # A categorical variable formed from all missing values will
                # have an empty categories list.
                index = pd.CategoricalIndex(
                    [np.nan], categories=categories, ordered=ordered,
                    dtype=index_dtype, name=name
                )
            else:
                index_values = pd.Categorical(categories[codes],
                                              categories=categories)
                # pd.Categorical uses -1 to encode a nan.  A code is -1, we
                # need to translate manually back to nan, otherwise Python will
                # interpret categories[-1] as "the last category" when it
                # should translate to nan.
                codes = np.array(codes)
                index_values[codes == -1] = np.nan
                index = pd.CategoricalIndex(
                    index_values,
                    categories=categories,
                    ordered=ordered,
                    dtype=index_dtype,
                    name=name,
                )
            return index

        except Exception as e:
            print("Error extracting CategoricalIndex from json payload.")
            print(f"payload: {payload}")
            print(e)
            raise e
