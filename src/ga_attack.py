import numpy as np
import numpy
import pandas as pd
from tqdm import tqdm
import itertools
import copy
import networkx as nx
import matplotlib.pyplot as plt
from alibi.utils.mapping import (
    ohe_to_ord as alibi_ohe_to_ord,
    ord_to_ohe as alibi_ord_to_ohe,
)

np.random.seed(555)

import matplotlib.pyplot

plt.style.use("ggplot")


# ***********************************************
# ***************** REMOOVE **********************
# ***********************************************
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
        count = count_similar(columns_inner[start_inner], columns_inner[end_inner - 1])
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


# ***********************************************
# ***************** REMOOVE **********************
# ***********************************************


def plot_graph(x_changes, threshold=1, verbose=False, figsize=(5,5), save=False):

    if len(x_changes) == 0:
        return

    names = np.asarray(list(x_changes.columns))
    x_changes = np.asarray(x_changes.values)
    G = nx.Graph()

    changes = {}
    for i in range(np.asarray(x_changes).shape[1]):
        if np.sum(np.abs(x_changes[:, i])) > 0:
            all_changes = list(zip(*np.unique(x_changes[:, i], return_counts=True)))
            changes[i] = [changed for changed in all_changes if changed[0] != 0]
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
        ax.set_facecolor('white')
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
            t.set_position((t.get_position()[0] + 0.1, t.get_position()[1] + 0.1))
            t.set_clip_on(False)
        plt.tight_layout()
        if save:
            plt.savefig("featchange_graph.png", type="png", dpi=600)
    else:
        print("No edges exist!")


def cal_pop_fitness(
    clf, y_target, x_in, pop, alpha, one_hots, n_class, alpha_0=0.6, beta_0=0.2
):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    pred = lambda x: predictproba_func(clf, x, one_hots)
    func_fitness = lambda x, x_in, y_target: (
        np.sum(np.power(pred(x) - np.repeat([y_target], len(x), axis=0), 2), axis=1)
        + alpha_0
        * alpha
        * np.sum(np.abs(np.nan_to_num(x_in) - np.nan_to_num(x)) > 0, axis=1)
        + beta_0
        * alpha
        * np.sum(np.abs(np.nan_to_num(x_in) - np.nan_to_num(x)), axis=1)
    )

    fitness = -1.0 * func_fitness(pop, x_in, y_target)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    idx_sorted = np.argsort(fitness)[::-1]
    parents = pop[idx_sorted[:num_parents]]
    return parents


def crossover(parents, offspring_size, blacklist=None):
    offspring = []
    if blacklist is None:
        blacklist = []
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]

        x_new = copy.deepcopy(parents[parent1_idx])
        x2 = copy.deepcopy(parents[parent2_idx])

        for i in range(offspring_size[1]):
            if i not in blacklist:
                if np.random.uniform() < 0.5:
                    x_new[i] = x2[i]

        offspring.append(x_new)

    offspring = np.asarray(offspring)
    return offspring


def mutation(offspring_crossover, X_train, num_mutations=1, blacklist=None, p_ind=None):
    if p_ind is None:
        p_ind = np.ones(X_train.shape[1]) / X_train.shape[1]
    p_ind = p_ind.reshape(-1)
    for idx in range(offspring_crossover.shape[0]):
        gene_idxs = np.random.choice(X_train.shape[1], size=num_mutations, p=p_ind)
        if blacklist is not None:
            gene_idxs = [ind for ind in gene_idxs if ind not in blacklist]
        for gene_idx in gene_idxs:
            offspring_crossover[idx, gene_idx] = np.random.choice(
                X_train[:, gene_idx], replace=False, size=1
            )

    return offspring_crossover


def predict_func(clf, x, ohe_vars_cat):
    x_in = x if np.ndim(x) > 1 else x.reshape(1, -1)
    if ohe_vars_cat is not None:
        x_in, _ = alibi_ord_to_ohe(x_in, ohe_vars_cat)
    return clf.predict(x_in)


def predictproba_func(clf, x, ohe_vars_cat):
    x_in = x if np.ndim(x) > 1 else x.reshape(1, -1)
    if ohe_vars_cat is not None:
        x_in, _ = alibi_ord_to_ohe(x_in, ohe_vars_cat)
    return clf.predict_proba(x_in)


def find_next_max(clf, x_in, ohe_vars_cat, target):

    ind_in = predict_func(clf, x_in, ohe_vars_cat)
    px = predictproba_func(clf, x_in, ohe_vars_cat)[0]
    if len(px) == 1:
        n_class = 2
        ind_target = 1 - ind_in[0]
    elif len(px) == 2:
        n_class = 2
        ind_target = 1 - ind_in[0]
    else:
        n_class = len(px)
        ind_target = np.argsort(px)[-2]

    if target is not None and target != ind_in:
        ind_target = target

    y_target = [0 for i in range(n_class)]
    y_target[ind_target] = 1

    return y_target, n_class, ind_target


def GA_Counterfactual(
    clf,
    x_train,
    x,
    target=None,
    sol_per_pop=20,
    num_parents_mating=5,
    num_generations=50,
    n_runs=10,
    num_mutations=1,
    black_list=None,
    cat_vars_ohe=None,
    beta=0.95,
    verbose=False,
    feature_names=None,
):

    # Onehot encoding and blacklist conversion
    x_in = copy.deepcopy(x)
    feature_names_new = None
    if feature_names is not None:
        feature_names_new = copy.deepcopy(feature_names)
    x_in = x_in if np.ndim(x_in) > 1 else x_in.reshape(1, -1)
    X_train = copy.deepcopy(x_train)
    ohe_vars_cat = None
    if cat_vars_ohe is not None:
        X_train, ohe_vars_cat = alibi_ohe_to_ord(X_train, cat_vars_ohe)
        x_in, _ = alibi_ohe_to_ord(x_in, cat_vars_ohe)
        x_in = x_in[0]
        col_dict, _ = _find_cols(cat_vars_ohe, x_train.shape[1], [])
        feature_names_new = (
            None
            if feature_names_new is None
            else _update_feature_name(feature_names_new, col_dict)
        )
        blacklist = []
        for i, ci in enumerate(col_dict):
            if isinstance(ci, int):
                ci = [ci]
            if len(set(ci).intersection(set(black_list))) > 0:
                blacklist.append(i)
                print(f"Removing {ci} form the list.")
    else:
        blacklist = black_list
    # ----------------------------------------

    y_target, n_class, ind_target = find_next_max(clf, x_in, ohe_vars_cat, target)
    best_outputs = []
    num_weights = len(x_in)
    pop_size = (sol_per_pop, num_weights)
    x_all = []
    x_changes = []
    x_sucess = []
    success = False

    gen_changes = 0
    check_success = 0
    for iter in tqdm(range(n_runs)):

        old_fitness = 0
        new_fitness = 1

        best_outputs = []
        alpha = 1.0

        if not success:
            gen_changes += 1

        # create the initial pool
        new_population = np.tile(x_in, (sol_per_pop, 1))
        for i in range(new_population.shape[0]):
            inds = np.random.randint(0, X_train.shape[1], gen_changes)
            if blacklist is not None:
                inds = [ind for ind in inds if ind not in blacklist]
            for ind in inds:
                new_population[i, ind] = np.random.choice(
                    X_train[:, ind], replace=True, size=1
                )

        success = False

        for generation in range(num_generations):

            # calculate the fitness to choose the best for next pool
            fitness = cal_pop_fitness(
                clf, y_target, x_in, new_population, alpha, ohe_vars_cat, n_class
            )
            best_outputs.append(np.max(fitness))

            best_output = new_population[numpy.where(fitness == numpy.max(fitness))[0]][
                0
            ]

            parents = select_mating_pool(new_population, fitness, num_parents_mating)

            # weighting the probability of chosen column by the amount of change in outcome
            # so the ones causing more change have higher probability of being selected again
            diff_x = (
                np.abs(np.nan_to_num(x_in) - np.nan_to_num(best_output)) > 0
            ).astype(int)
            delta = (
                predictproba_func(clf, best_output, ohe_vars_cat)[0][ind_target]
                - predictproba_func(clf, x_in, ohe_vars_cat)[0][ind_target]
            )
            delta = np.max([0.0, delta])
            p_ind = np.exp(-delta * diff_x / 0.5)
            p_ind /= p_ind.sum()

            # loosening the constraint if the outcome is not changed
            if predict_func(clf, best_output, ohe_vars_cat) == predict_func(
                clf, x_in, ohe_vars_cat
            ):
                if check_success > 1:  # should 1 be a parameter too?
                    alpha *= beta
                    if verbose:
                        print(f"alpha is changing to {alpha}.")
                    check_success = 0
                check_success += 1
            else:
                old_fitness = new_fitness
                new_fitness = np.max(fitness)

                # early stop if the outcome is changed and no further changes are happening
                if new_fitness == old_fitness:
                    if verbose:
                        print("Early Stopping....")
                    break

            # Generating next generation using crossover and mutation
            offspring_crossover = crossover(
                parents,
                offspring_size=(pop_size[0] - parents.shape[0], num_weights),
                blacklist=blacklist,
            )
            offspring_mutation = mutation(
                offspring_crossover,
                X_train,
                num_mutations=num_mutations,
                blacklist=blacklist,
                p_ind=p_ind,
            )
            new_population[0 : parents.shape[0], :] = parents
            new_population[parents.shape[0] :, :] = offspring_mutation

        fitness = cal_pop_fitness(
            clf, y_target, x_in, new_population, alpha, ohe_vars_cat, n_class
        )
        best_match_idx = numpy.where(fitness == numpy.max(fitness))[0]

        # choose the one with the best score as the final result
        x_cf = new_population[best_match_idx[0], :]
        diff = (np.nan_to_num(x_in) - np.nan_to_num(x_cf)).reshape(-1)
        ndiff = (np.abs(diff) > 0).sum()

        x_all.append(x_cf)

        if predict_func(clf, x_in, ohe_vars_cat) != predict_func(
            clf, x_cf, ohe_vars_cat
        ):
            x_sucess.append(x_cf.reshape(-1))
            x_changes.append(diff.reshape(-1))
            success = True

        if verbose:
            print("Difference: ", -1 * diff, ndiff)
            print("The counterfactual example: ", x_cf)
            print(
                "Predictions: ",
                predict_func(clf, x_in, ohe_vars_cat),
                predict_func(clf, new_population[best_match_idx[0], :], ohe_vars_cat),
            )
            matplotlib.pyplot.figure()
            matplotlib.pyplot.plot(best_outputs)
            matplotlib.pyplot.xlabel("Iteration")
            matplotlib.pyplot.ylabel("Fitness")
            matplotlib.pyplot.show()

    # x_changes contains the features that have changed in the code
    if len(x_changes) > 0:
        x_changes = np.asarray(x_changes)
        if feature_names_new is None:
            feature_names_new = [str(i) for i in range(x_changes.shape[1])]
        x_changes = pd.DataFrame(data=x_changes, columns=feature_names_new)
    return x_all, x_changes, x_sucess, ohe_vars_cat, n_class


class GAdvExample(object):
    def __init__(
        self,
        cat_vars_ohe=None,
        feature_names=None,
        sol_per_pop=30,
        num_parents_mating=10,
        num_generations=100,
        n_runs=10,
        black_list=None,
        beta=0.95,
        verbose=False,
        target=None,
    ):
        # TODO: being able to assign black_list with names.
        # TODO: data type conversion

        self.cat_vars_ohe = cat_vars_ohe
        self.feature_names = feature_names
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.n_runs = n_runs
        self.num_mutations = 1
        if black_list is None:
            self.black_list = []
        else:
            self.black_list = black_list

        self.beta = beta
        self.verbose = verbose
        self.target = target

    def attack(self, estimator, x, x_train):
        x_in = copy.deepcopy(x)
        x_in = x_in if np.ndim(x_in) > 1 else x_in.reshape(1, -1)
        x_all, x_changes, x_sucess, self.ohe_vars_cat, n_class = GA_Counterfactual(
            estimator,
            x_train,
            x_in,
            target=self.target,
            sol_per_pop=self.sol_per_pop,
            num_parents_mating=self.num_parents_mating,
            cat_vars_ohe=self.cat_vars_ohe,
            num_generations=self.num_generations,
            n_runs=self.n_runs,
            num_mutations=self.num_mutations,
            black_list=self.black_list,
            verbose=self.verbose,
            beta=self.beta,
            feature_names=self.feature_names,
        )

        self.results = None
        if len(x_sucess) > 0:
            if self.ohe_vars_cat is None:
                x_sucess_ord = np.asarray(x_sucess)
            else:
                x_sucess_ord, _ = alibi_ord_to_ohe(
                    np.asarray(x_sucess), self.ohe_vars_cat
                )

            df_in = pd.DataFrame(data=x_in, columns=self.feature_names)
            df_adv = pd.DataFrame(data=x_sucess_ord, columns=self.feature_names)

            for i in range(n_class):
                df_in[f"P{i}"] = estimator.predict_proba(x_in)[0][i]

                df_adv[f"P{i}"] = estimator.predict_proba(x_sucess_ord)[:, i]

            self.results = pd.concat([df_in, df_adv], axis=0).reset_index(drop=True)
            self.results.drop_duplicates(inplace=True)

            def highlight_change(s):
                is_changed = s != s.iloc[0]
                return ["background-color: darkorange" if v else "" for v in is_changed]

            self.results = self.results.style.apply(highlight_change)

        return x_all, x_changes, x_sucess
