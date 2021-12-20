"""
The functions are from: https://github.com/SeldonIO/alibi/blob/master/alibi/utils/mapping.py
"""
import itertools
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def ord_to_ohe(
    X_ord: np.ndarray, cat_vars_ord: dict
) -> Tuple[np.ndarray, dict]:
    """
    Convert ordinal to one-hot encoded variables.
    Parameters
    ----------
    X_ord
        Data with mixture of ordinal encoded and numerical variables.
    cat_vars_ord
        Dict with as keys the categorical columns and as values
        the number of categories per categorical variable.
    Returns
    -------
    One-hot equivalent of ordinal encoded data and dict with categorical columns and number of categories.
    """
    n, cols = X_ord.shape
    ord_vars_keys = list(cat_vars_ord.keys())
    X_list = []
    c = 0
    k = 0
    cat_vars_ohe = {}
    while c < cols:
        if c in ord_vars_keys:
            v = cat_vars_ord[c]
            X_ohe_c = np.zeros((n, v), dtype=np.float32)
            X_ohe_c[np.arange(n), X_ord[:, c].astype(int)] = 1.0
            cat_vars_ohe[k] = v
            k += v
            X_list.append(X_ohe_c)
        else:
            X_list.append(X_ord[:, c].reshape(n, 1))
            k += 1
        c += 1
    X_ohe = np.concatenate(X_list, axis=1)
    return X_ohe, cat_vars_ohe


def ohe_to_ord(
    X_ohe: np.ndarray, cat_vars_ohe: dict
) -> Tuple[np.ndarray, dict]:
    """
    Convert one-hot encoded variables to ordinal encodings.
    Parameters
    ----------
    X_ohe
        Data with mixture of one-hot encoded and numerical variables.
    cat_vars_ohe
        Dict with as keys the first column index for each one-hot encoded categorical variable
        and as values the number of categories per categorical variable.
    Returns
    -------
    Ordinal equivalent of one-hot encoded data and dict with categorical columns and number of categories.
    """
    n, cols = X_ohe.shape
    ohe_vars_keys = list(cat_vars_ohe.keys())
    X_list = []  # type: List
    c = 0
    cat_vars_ord = {}
    while c < cols:
        if c in ohe_vars_keys:
            v = cat_vars_ohe[c]
            X_ohe_c = X_ohe[:, c : c + v]
            assert int(np.sum(X_ohe_c, axis=1).sum()) == n
            X_ord_c = np.argmax(X_ohe_c, axis=1)
            cat_vars_ord[len(X_list)] = v
            X_list.append(X_ord_c.reshape(n, 1))
            c += v
            continue
        X_list.append(X_ohe[:, c].reshape(n, 1))
        c += 1
    X_ord = np.concatenate(X_list, axis=1)
    return X_ord, cat_vars_ord


def _findSubStr(X, Y, m, n):
    """
    Helper function to find the longest similar substring in two strings.
    """
    LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]
    result = 0

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                LCSuff[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                result = max(result, LCSuff[i][j])
            else:
                LCSuff[i][j] = 0
    return result


def _update_feature_name(in_names, col_dict):
    """
    Finds the longest substring shared in the one-hot encoded column names.
    """
    feature_names = []
    for i, (k, v) in enumerate(col_dict.items()):
        if isinstance(v, int):
            feature_names.append(in_names[v])
        elif isinstance(v, (list, np.ndarray)):
            if len(v) == 1:
                feature_names.append(in_names[v[0]])
            else:
                name_lists = [in_names[vi] for vi in v]
                common_substring = name_lists[0]
                for name_list in name_lists[1:]:
                    common_substring = common_substring[
                        : _findSubStr(
                            common_substring,
                            name_list,
                            len(common_substring),
                            len(name_list),
                        )
                    ]
                feature_names.append(common_substring[:-1])

    return feature_names


def _find_cols(cat_vars_ohe, length, categorical_columns):
    """
    Convert the input one-hot encoding dictionary to a
    dictionary that can be used for ordinal to one-hot conversion.
    In addition returns a list of categorical columns including ones
    defined in input and

    Args:
        cat_vars_ohe (dict): Keys are the first column index for each one-hot encoded
            categorical variable and values are the number of categories per categorical
            variable.
        length (int): Total number of columns
        categorical_columns (list of int): list of the index of the categorical (non-one
            hot encoded) columns.
    Returns:
        tuple:
            dict:
                Keys are the new column indexes and values are the values of the
                non-one hot encoded categorical variables (if a list) or the old column
                index.
            list:
                list of bool values showing if the column is categorical
    """
    if categorical_columns is None:
        categorical_columns = []

    list2d = [list(range(k, k + v)) for k, v in cat_vars_ohe.items()]
    merged = list(itertools.chain(*list2d))

    i, j, k = 0, 0, 0
    col_dict = {}
    categorical = []

    while i < length:
        if i not in merged:
            col_dict[j] = i
            if i in categorical_columns:
                categorical.append(True)
            else:
                categorical.append(False)
            i += 1
            j += 1
        else:
            col_dict[j] = list2d[k]

            i += len(list2d[k])
            j += 1
            k += 1
            categorical.append(True)
    return col_dict, categorical


def _find_similar_columns(feature_names, sep="_", min_dist=1, data=None):
    """
    Helper function to be used by `create_onehot_map` to create one-hot map dictionary.
    """
    if data is not None:
        if not isinstance(data, np.ndarray):
            raise Exception("Data should be an ndarray.")
        if data.shape[1] != len(feature_names):
            raise Exception(
                "Data should have the same number of columns as the length of feature_names."
            )
        else:
            binary_cols = []
            for i, fn in enumerate(feature_names):
                sxli = set(data[:, i])
                if sxli == {0, 1} or sxli == {0} or sxli == {1}:
                    binary_cols.append(i)
    else:
        binary_cols = range(len(feature_names))

    def count_similar(x1, x2):
        if len(x2) > len(x1):
            x2, x1 = x1, x2
        count = 0
        for i in range(len(x2)):
            if x2[i] == x1[i]:
                count += 1
            else:
                break
        return count

    def find_name(columns_inner, start_inner, end_inner):
        count = count_similar(
            columns_inner[start_inner], columns_inner[end_inner - 1]
        )
        return "_".join(columns_inner[start_inner][:count])

    columns = []
    start = 0
    end = 1
    col_all = []
    names = []
    for ci in feature_names:
        columns.append(ci.split(sep))

    while end < len(columns):
        if (
            count_similar(columns[start], columns[end]) >= min_dist
            and end in binary_cols
        ):
            end += 1
        else:
            if end - start > 1:
                if data is not None:
                    if np.all(data[:, range(start, end)].sum(1) == 1):
                        col_all.append(list(range(start, end)))
                        names.append(find_name(columns, start, end))
                else:
                    col_all.append(list(range(start, end)))
                    names.append(find_name(columns, start, end))
            start = end
            end += 1

    if end - start > 1:
        if data is not None:
            if np.all(data[:, range(start, end)].sum(1) == 1):
                col_all.append(list(range(start, end)))
                names.append(find_name(columns, start, end))
        else:
            col_all.append(list(range(start, end)))
            names.append(find_name(columns, start, end))

    return col_all, names


def create_onehot_map(feature_names, sep="_", min_dist=1, data=None):
    """
    Helper function to create onehot encoding mapping dictionary from feature_names and data
    The onehot encoded names are assumed to follow scikit-learn/pandas onehot encoding format
    e.g. all have the same name with values being separated by `_`.

    The code uses the names and checks the data values to make sure only one of the values is
    one (as expected from one-hot encoded data).

    In addition, if there are more than one `_` in the names, you may need to play with `min_dist`
    (e.g., increase it) to get a better detection.

    Args:
        feature_names (list of str): list of the name of the features.
        sep (str): separation character. Defaults to `_`.
        min_dist (int): the minimum number of similarity (separeted by `sep`) needed to be
            considered as the same feature.
        data (:class:`~numpy.ndarray`, optional): if given will be used to check the one-hot
            encoding correctness (summation of each set of feature in each row must be `1`).
    """
    col_all, names = _find_similar_columns(feature_names, sep, min_dist, data)

    onehot_map = {}

    for c in col_all:
        onehot_map[c[0]] = len(c)

    return names, onehot_map


def plot_graph(
    x_changes, threshold=1, verbose=False, figsize=(5, 5), save=False
):

    if len(x_changes) == 0:
        return

    names = np.asarray(list(x_changes.columns))
    x_changes = np.asarray(x_changes.values)
    G = nx.Graph()

    changes = {}
    for i in range(np.asarray(x_changes).shape[1]):
        if np.sum(np.abs(x_changes[:, i])) > 0:
            all_changes = list(
                zip(*np.unique(x_changes[:, i], return_counts=True))
            )
            changes[i] = [
                changed for changed in all_changes if changed[0] != 0
            ]
            if verbose:
                print(names[i], changes[i])

    XR = []
    for i in range(x_changes.shape[0]):
        nz = np.where(x_changes[i, :] != 0)[0]
        for r in itertools.combinations(nz, 2):
            if r:
                XR.append(r)

    edges = {}
    for t in XR:
        ts = tuple(t)
        edges[ts] = 1 + edges.get(ts, 0)

    if verbose:
        print(edges)

    for e in edges.keys():
        G.add_weighted_edges_from([(*e, edges[e])])

    # pos = nx.spring_layout(G)  # positions for all nodes
    pos = nx.circular_layout(G)

    if len(XR) > 0:
        # figsize is intentionally set small to condense the graph
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")
        margin = 0.33
        fig.subplots_adjust(margin, margin, 1.0 - margin, 1.0 - margin)
        ax.axis("equal")

        # nodes
        nx.draw_networkx_nodes(G, pos, node_size=500)
        description = nx.draw_networkx_labels(
            G, pos, {n: names[n] for n in G.nodes}, font_size=12
        )

        for e in edges.keys():
            if edges[e] >= threshold:
                nx.draw_networkx_edges(
                    G, pos, edgelist=[e], width=10 * edges[e] / len(x_changes)
                )

        r = fig.canvas.get_renderer()
        trans = plt.gca().transData.inverted()
        for node, t in description.items():
            bb = t.get_window_extent(renderer=r)
            bbdata = bb.transformed(trans)
            t.set_position(
                (t.get_position()[0] + 0.1, t.get_position()[1] + 0.1)
            )
            t.set_clip_on(False)
        plt.tight_layout()
        if save:
            plt.savefig("featchange_graph.png", type="png", dpi=600)
    else:
        print("No edges exist!")
