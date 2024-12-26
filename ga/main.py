from rust_reversi import Arena
import sys
import numpy as np
import copy
import tqdm

CHAMPION_PLAYER = "players/champion_player.py"
TMP_MATRIX_PLAYER = "players/tmp_matrix_player.py"

MATRIX_DEPTH = 1

BASE_MATRIX = [
    [50, -10, 11, 6, 6, 11, -10, 50],
    [-10, -15, 1, 2, 2, 1, -15, -10],
    [11, 1, 1, 1, 1, 1, 1, 11],
    [6, 2, 1, 3, 3, 1, 2, 6],
    [6, 2, 1, 3, 3, 1, 2, 6],
    [11, 1, 1, 1, 1, 1, 1, 11],
    [-10, -15, 1, 2, 2, 1, -15, -10],
    [50, -10, 11, 6, 6, 11, -10, 50],
]


class GeneticOptimizer:
    def __init__(
        self,
        population_size=20,
        n_games=500,
        initial_range=20,
        mutation_rate=0.3,
        mutation_range=10,
        n_generations=20,
        base_rate=0.2,
    ):
        self.population_size = population_size
        self.n_games = n_games
        self.python = sys.executable
        self.matrix_player = [
            self.python,
            TMP_MATRIX_PLAYER,
            str(MATRIX_DEPTH),
        ]
        self.champion_player = [self.python, CHAMPION_PLAYER, str(MATRIX_DEPTH)]
        self.mutation_rate = mutation_rate
        self.mutation_range = mutation_range
        self.n_generations = n_generations

        self.population = []
        # Initialize population based on base matrix
        for _ in range(int(population_size * base_rate)):
            noise = np.random.randint(-mutation_range, mutation_range, (8, 8))
            mask = np.random.random((8, 8)) < mutation_rate
            matrix = np.array(BASE_MATRIX) + mask * noise
            self.population.append(matrix.astype(int))
        # Initialize population with random matrices
        while len(self.population) < population_size:
            matrix = np.random.randint(-initial_range, initial_range, (8, 8))
            self.population.append(matrix)

        # champion player with base matrix
        with open(TMP_MATRIX_PLAYER, "r") as f:
            lines = f.readlines()
        matrix_str = "EVAL_MATRIX = [\n"
        for row in BASE_MATRIX:
            matrix_str += "    [" + ", ".join(map(str, row)) + "],\n"
        matrix_str += "]\n"
        for i, line in enumerate(lines):
            if line.startswith("EVAL_MATRIX"):
                end_idx = i
                while not lines[end_idx].strip().endswith("]"):
                    end_idx += 1
                lines[i : end_idx + 1] = [matrix_str]
                break
        with open(CHAMPION_PLAYER, "w") as f:
            f.writelines(lines)

    def evaluate_fitness(self, matrix):
        self._save_matrix(matrix)
        arena = Arena(self.champion_player, self.matrix_player, show_progress=False)
        arena.play_n(self.n_games)
        p1_win, p2_win, draw = arena.get_stats()
        return (p2_win + 0.5 * draw - p1_win) / self.n_games

    def _save_matrix(self, matrix, file=TMP_MATRIX_PLAYER):
        with open(file, "r") as f:
            lines = f.readlines()

        matrix_str = "EVAL_MATRIX = [\n"
        for row in matrix:
            matrix_str += "    [" + ", ".join(map(str, row)) + "],\n"
        matrix_str += "]\n"

        for i, line in enumerate(lines):
            if line.startswith("EVAL_MATRIX"):
                end_idx = i
                while not lines[end_idx].strip().endswith("]"):
                    end_idx += 1
                lines[i : end_idx + 1] = [matrix_str]
                break

        with open(file, "w") as f:
            f.writelines(lines)

    def crossover(self, parent1, parent2):
        child = np.zeros((8, 8), dtype=int)
        for i in range(8):
            for j in range(8):
                child[i][j] = int((parent1[i][j] + parent2[i][j]) / 2)
        rot90 = np.rot90(child)
        rot180 = np.rot90(rot90)
        rot270 = np.rot90(rot180)
        flip = np.flip(child, axis=1)
        flip_rot90 = np.rot90(flip)
        flip_rot180 = np.rot90(flip_rot90)
        flip_rot270 = np.rot90(flip_rot180)
        child = (
            child
            + rot90
            + rot180
            + rot270
            + flip
            + flip_rot90
            + flip_rot180
            + flip_rot270
        ) / 8
        return child.astype(int)

    def mutate(self, matrix):
        mask = np.random.random((8, 8)) < self.mutation_rate
        mutations = np.random.randint(-self.mutation_range, self.mutation_range, (8, 8))
        matrix = matrix + mask * mutations
        return matrix.astype(int)

    def evolve(self):
        for gen in tqdm.tqdm(range(self.n_generations)):
            fitness_scores = [
                self.evaluate_fitness(matrix)
                for matrix in tqdm.tqdm(self.population, leave=False)
            ]

            # update champion player
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > 0.1:
                self._save_matrix(self.population[best_idx], CHAMPION_PLAYER)

            new_population = []
            elite_indices = np.argsort(fitness_scores)[-2:]
            for idx in elite_indices:
                new_population.append(copy.deepcopy(self.population[idx]))

            while len(new_population) < self.population_size:
                tournament_indices1 = np.random.choice(
                    len(self.population), size=4, replace=False
                )
                parent1_idx = tournament_indices1[
                    np.argmax([fitness_scores[i] for i in tournament_indices1])
                ]
                tournament_indices2 = np.random.choice(
                    len(self.population), size=4, replace=False
                )
                parent2_idx = tournament_indices2[
                    np.argmax([fitness_scores[i] for i in tournament_indices2])
                ]

                child = self.crossover(
                    self.population[parent1_idx], self.population[parent2_idx]
                )
                child = self.mutate(child)
                new_population.append(child)

            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            print(f"Generation {gen + 1}")
            print(f"Best/Avg Fitness: {best_fitness:.3f}/{avg_fitness:.3f}")
            print("Best Matrix:")
            print(self.population[np.argmax(fitness_scores)])
            print()
            optimizer._save_matrix(self.population[np.argmax(fitness_scores)])
            self.population = new_population


if __name__ == "__main__":
    optimizer = GeneticOptimizer()
    optimizer.evolve()
