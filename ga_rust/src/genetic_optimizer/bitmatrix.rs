use rand::Rng;

use crate::fitness_calculator::bitmatrix::{EvaluatorType, FitnessCalculator};
use crate::genetic_evaluators::bitmatrix::GeneticBitMatrixEvaluator;
use crate::genetic_optimizer::OptimizerConfig;

pub struct BitMatrixOptimizer<const N: usize> {
    population: Vec<GeneticBitMatrixEvaluator<N>>,
    fitness_calculator: Box<dyn FitnessCalculator<N>>,
    generation: usize,
    config: OptimizerConfig,
}

impl<const N: usize> BitMatrixOptimizer<N> {
    pub fn new(
        fitness_calculator: Box<dyn FitnessCalculator<N>>,
        config: OptimizerConfig,
    ) -> BitMatrixOptimizer<N> {
        let mut population = Vec::new();
        for _ in 0..config.population_size {
            population.push(GeneticBitMatrixEvaluator::<N>::new_from_random());
        }
        BitMatrixOptimizer {
            population,
            fitness_calculator,
            generation: 0,
            config,
        }
    }

    fn evaluate_fitness(&self) -> Vec<f64> {
        let evaluators = self
            .population
            .iter()
            .map(|evaluator| EvaluatorType::BitMatrix(evaluator.to_evaluator()))
            .collect::<Vec<_>>();
        self.fitness_calculator.calculate_fitness(evaluators)
    }

    fn select(&self, fitnesses: Vec<f64>) -> Vec<GeneticBitMatrixEvaluator<N>> {
        let mut rng = rand::thread_rng();
        let mut selected = Vec::new();
        for i in 0..self.config.population_size {
            let mut best_index = i;
            let mut best_fitness = fitnesses[i];
            for _i in 0..self.config.tournament_size {
                let index = rng.gen_range(0..self.config.population_size);
                if fitnesses[index] > best_fitness {
                    best_index = index;
                    best_fitness = fitnesses[index];
                }
            }
            selected.push(self.population[best_index].clone());
        }
        selected
    }

    fn evolve(&mut self, fitnesses: Vec<f64>) {
        let selected = self.select(fitnesses);
        let mut new_population = Vec::new();
        for i in 0..self.config.population_size {
            let mut rng = rand::thread_rng();
            let parent1 = match rng.gen_bool(self.config.mutation_rate) {
                true => &selected[i].mutate(),
                false => &selected[i],
            };
            match rng.gen_bool(self.config.crossover_rate) {
                true => {
                    let parent2 = selected[rng.gen_range(0..self.config.population_size)].clone();
                    new_population.push(parent1.crossover(&parent2));
                }
                false => new_population.push(parent1.clone()),
            }
        }
        self.population = new_population;
    }

    pub fn optimize(&mut self) -> GeneticBitMatrixEvaluator<N> {
        let mut global_best_indivisual = GeneticBitMatrixEvaluator::<N>::new_from_random();
        let mut global_best_fitness = 0.0;
        for _ in 0..self.config.max_generations {
            let mut best_index = 0;
            let mut best_fitness = 0.0;
            let fitnesses = self.evaluate_fitness();
            for (i, &fitness) in fitnesses.iter().enumerate() {
                if fitness > best_fitness {
                    best_index = i;
                    best_fitness = fitness;
                }
            }
            if best_fitness > global_best_fitness {
                global_best_indivisual = self.population[best_index].clone();
                global_best_fitness = best_fitness;
            }
            println!(
                "Generation: {}, Best Fitness: {}",
                self.generation, best_fitness
            );
            println!("Best Individual: {:?}", self.population[best_index]);
            self.evolve(fitnesses);
            self.generation += 1;
        }
        println!("Global Best Fitness: {}", global_best_fitness);
        println!("Global Best Individual: {:?}", global_best_indivisual);
        global_best_indivisual
    }
}
