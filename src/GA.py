import hybrid_model as hbm
import data_proc as data
import pygad

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],  
    'activation': ['relu', 'tanh'],  
    'solver': ['adam', 'sgd'],  
    'learning_rate_init': [0.001, 0.01],  
    'max_iter': [100, 300]  
}

def params_from_solu(s):
    hls = param_grid['hidden_layer_sizes'][int(s[0] % len(param_grid['hidden_layer_sizes']))]
    act = param_grid['activation'][int(s[1] % len(param_grid['activation']))]
    sol = param_grid['solver'][int(s[0] % len(param_grid['solver']))]
    lri = param_grid['learning_rate_init'][int(s[0] % len(param_grid['learning_rate_init']))]
    mxi = param_grid['max_iter'][int(s[0] % len(param_grid['max_iter']))]
    return (hls, act, sol, lri, mxi)


class MLP_GA():
    def __init__(self):
        self.y_train = None
        self.y_test = None
        self.X_train = None
        self.X_test = None
        self.best_params = None
        self.initialized = False

    def init_ga(self, df):
        train_df, test_df = data.train_test_split_df(df, size=0.2, random_state=42)

        cbf_model = hbm.CBFModel(train_df, decision_threshold=0.005)
        cbf_model.fit()

        cb_svd = hbm.CBModel(train_df, n_components=700)
        cb_svd.fit()

        cbcf_train_df = data.create_cbcf_df(cbf_model, cb_svd, train_df)
        cbcf_test_df = data.create_cbcf_df(cbf_model, cb_svd, test_df)

        self.y_train = cbcf_train_df['rating'].values
        self.y_test = cbcf_test_df['rating'].values
        self.X_train = cbcf_train_df.drop(columns=['rating']).values
        self.X_test = cbcf_test_df.drop(columns=['rating']).values
        self.best_params = None
        self.initialized = True

    def fitness_func(self, ga_instance, solution, solution_idx):
        (hls, act, sol, lri, mxi) = params_from_solu(solution)
        mlp = MLPClassifier(hidden_layer_sizes=hls, activation=act, solver=sol, learning_rate_init=lri, max_iter=mxi, random_state=42)
        mlp.fit(self.X_train, self.y_train)
        y_pred = mlp.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        return acc

    def train_best_mlp(self):
        if self.best_params is None:
            raise Exception("Run the GA before training the best MLP.")
        (hls, act, sol, lri, mxi) = self.best_params
        mlp = MLPClassifier(hidden_layer_sizes=hls, activation=act, solver=sol, learning_rate_init=lri, max_ite=mxi, random_state=42)
        mlp.fit(X_train, y_train)
        return mlp

    # Returns best parameters in form (hls, act, sol, lri, mxi)
    def run_ga(self):
        if not self.initialized:
            raise Exception("GA not properly initialized.")

        fitness_function = self.fitness_func
        num_generations = 3            # def 50
        num_parents_mating = 4          # def 4
        sol_per_pop = 8                 # def 8
        num_genes = len(param_grid)
        init_range_low = -2             # def -2
        init_range_high = 5             # def 5
        parent_selection_type = "sss"   # def 'sss'
        keep_parents = 1                # def 1
        crossover_type = "single_point" # def 'single_point'
        mutation_type = "random"        # def 'random'
        mutation_percent_genes = 10     # def 10

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=init_range_low,
                               init_range_high=init_range_high,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_percent_genes=mutation_percent_genes)
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Parameters of the best solution : {solution}".format(solution=solution))
        print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

        best_params = params_from_solu(solution)
        print("Best MLP parameters: ", best_params)
        self.best_params = best_params
        return best_params
