import copy

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import _find_cols, _update_feature_name
from .utils import ohe_to_ord as alibi_ohe_to_ord
from .utils import ord_to_ohe as alibi_ord_to_ohe

np.random.seed(555)


def find_samples(data_df, data_qt, sample, ind_feature, keep=None):
    ind_feature_copy = copy.deepcopy(ind_feature)
    if not isinstance(ind_feature, (list, tuple)):
        ind_feature = [ind_feature]
    if keep is not None:
        if len(keep) > 0:
            ind_feature += keep
    else:
        keep = []

    other_features = [
        i for i in range(data_qt.shape[1]) if i not in ind_feature
    ]

    logic = (
        np.sum(data_qt[:, other_features] == sample[0, other_features], axis=1)
        > len(other_features) * 2 / 3
    )

    logic_keep = np.ones_like(logic)
    for keepi in keep:
        logic_keep = np.logical_and(
            logic_keep, data_qt[:, keepi] == sample[0, keepi]
        )

    if len(keep) > data_qt.shape[1] / 3:
        logic = logic_keep
    else:
        logic = np.logical_and(logic, logic_keep)

    inds_similar = np.where(logic)[0]

    values = data_df[inds_similar, ind_feature_copy]

    if len(values) == 0 and len(keep) > 0:
        inds_similar = np.where(logic_keep)[0]
        values = data_df[inds_similar, ind_feature_copy]

    if len(values) == 0 and len(keep) > 0:
        logic = (
            np.sum(
                data_qt[:, other_features] == sample[0, other_features], axis=1
            )
            > len(other_features) * 2 / 3
        )
        inds_similar = np.where(logic)[0]
        values = data_df[inds_similar, ind_feature_copy]

    return values.flatten()


def find_qtval(sample, bins, ind_feature):
    bini = bins[ind_feature]
    vali = sample[0, ind_feature]

    if len(bini) > 0:
        for i in range(len(bini) - 1):
            if i == 0:
                if bini[i + 1] >= vali >= bini[i]:
                    return i
            else:
                if bini[i + 1] >= vali > bini[i]:
                    return i
    else:
        return vali


def df2qt(sample, bins):
    sample_qt = sample.copy()
    for i in range(sample.shape[1]):
        vali = find_qtval(sample, bins, i)
        sample_qt[0, i] = vali
    return sample_qt


def find_closest(replacement_values, crossover_val):

    delta = np.abs(np.asarray(replacement_values) - crossover_val)
    value = replacement_values[np.argsort(delta)[0]]

    return [value]


def create_qt(data_df, nunique=5):
    data_qt = data_df.copy()
    bins = {}
    for i, ci in enumerate(data_qt.columns):
        if data_df[ci].nunique() > nunique:
            data_qt[ci], bins[i] = pd.qcut(
                data_df[ci],
                nunique,
                labels=False,
                duplicates="drop",
                retbins=True,
            )
        else:
            bins[i] = []

    return data_qt, bins


def gibbs_sampling(
    data_df,
    data_qt,
    bins,
    sample_in,
    ind_features,
    niter=5,
    crossover_val=None,
):
    if not isinstance(ind_features, (list, tuple)):
        ind_features = [ind_features]

    sample = sample_in.reshape(1, -1).copy()

    if crossover_val is not None:
        assert len(crossover_val) == len(
            ind_features
        ), "crossover_val and ind_features should have same length."

    if len(ind_features) == 1:
        niter = 1
    for i in range(niter):
        for i_ind_feature, ind_feature in enumerate(ind_features):
            sample_qt = df2qt(sample, bins)

            ind_features_ = [
                indf for indf in ind_features if indf != ind_feature
            ]

            replacement_values = find_samples(
                data_df, data_qt, sample_qt, ind_feature, keep=ind_features_
            )
            replacement_values = [
                xi for xi in replacement_values if xi != sample[0, ind_feature]
            ]

            if len(replacement_values) == 0:
                replacement_values = [sample[0, ind_feature]]

            if len(replacement_values) > 1 and crossover_val is not None:
                replacement_values = find_closest(
                    replacement_values, crossover_val[i_ind_feature]
                )

            sample[0, ind_feature] = np.random.choice(replacement_values, 1)[0]

    return sample


def plot_graph(x_changes, threshold=1, verbose=False, figsize=(5, 5)):

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
    else:
        print("No edges exist!")


def cal_pop_fitness(
    clf, y_target, x_in, pop, alpha, one_hots, n_class, alpha_0=0.6, beta_0=0.2
):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    pred = lambda x: predictproba_func(clf, x, one_hots)
    func_fitness = lambda x, x_in, y_target: (
        np.sum(
            np.power(pred(x) - np.repeat([y_target], len(x), axis=0), 2),
            axis=1,
        )
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


def crossover(parents, X_train, data_qt, bins, offspring_size, blacklist=None):
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

        inds = []
        for i in range(offspring_size[1]):
            if i not in blacklist:
                if np.random.uniform() < 0.5:
                    inds.append(i)
        x_new = gibbs_sampling(
            X_train,
            data_qt,
            bins,
            x_new,
            inds,
            niter=3,
            crossover_val=x2[inds],
        )[0]
        offspring.append(x_new)

    offspring = np.asarray(offspring)
    return offspring


def mutation(
    offspring_crossover,
    X_train,
    data_qt,
    bins,
    num_mutations=1,
    blacklist=None,
    p_ind=None,
):
    if p_ind is None:
        p_ind = np.ones(X_train.shape[1]) / X_train.shape[1]
    p_ind = p_ind.reshape(-1)
    for idx in range(offspring_crossover.shape[0]):
        gene_idxs = np.random.choice(
            X_train.shape[1], size=num_mutations, p=p_ind
        )
        if blacklist is not None:
            gene_idxs = [ind for ind in gene_idxs if ind not in blacklist]

        offspring_crossover[idx, :] = gibbs_sampling(
            X_train,
            data_qt,
            bins,
            offspring_crossover[idx, :],
            gene_idxs,
            niter=3,
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
    # create the quantized data

    data_qt, bins = create_qt(
        pd.DataFrame(data=X_train, columns=feature_names_new)
    )
    data_qt = data_qt.values

    # ----------------------------------------

    y_target, n_class, ind_target = find_next_max(
        clf, x_in, ohe_vars_cat, target
    )
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

            new_population[i, :] = gibbs_sampling(
                X_train, data_qt, bins, new_population[i, :], inds, niter=3
            )

        success = False

        for generation in range(num_generations):

            # calculate the fitness to choose the best for next pool
            fitness = cal_pop_fitness(
                clf,
                y_target,
                x_in,
                new_population,
                alpha,
                ohe_vars_cat,
                n_class,
            )
            best_outputs.append(np.max(fitness))

            best_output = new_population[
                numpy.where(fitness == numpy.max(fitness))[0]
            ][0]

            parents = select_mating_pool(
                new_population, fitness, num_parents_mating
            )

            # weighting the probability of chosen column by the amount of change in outcome
            # so the ones causing more change have higher probability of being selected again
            diff_x = (
                np.abs(np.nan_to_num(x_in) - np.nan_to_num(best_output)) > 0
            ).astype(int)
            delta = (
                predictproba_func(clf, best_output, ohe_vars_cat)[0][
                    ind_target
                ]
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
                X_train,
                data_qt,
                bins,
                offspring_size=(pop_size[0] - parents.shape[0], num_weights),
                blacklist=blacklist,
            )
            offspring_mutation = mutation(
                offspring_crossover,
                X_train,
                data_qt,
                bins,
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
                predict_func(
                    clf, new_population[best_match_idx[0], :], ohe_vars_cat
                ),
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
        (
            x_all,
            x_changes,
            x_sucess,
            self.ohe_vars_cat,
            n_class,
        ) = GA_Counterfactual(
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
            df_adv = pd.DataFrame(
                data=x_sucess_ord, columns=self.feature_names
            )

            for i in range(n_class):
                df_in[f"P{i}"] = estimator.predict_proba(x_in)[0][i]

                df_adv[f"P{i}"] = estimator.predict_proba(x_sucess_ord)[:, i]

            self.results = pd.concat([df_in, df_adv], axis=0).reset_index(
                drop=True
            )
            self.results.drop_duplicates(inplace=True)

            def highlight_change(s):
                is_changed = s != s.iloc[0]
                return [
                    "background-color: darkorange" if v else ""
                    for v in is_changed
                ]

            self.results = self.results.style.apply(highlight_change)

        return x_all, x_changes, x_sucess
