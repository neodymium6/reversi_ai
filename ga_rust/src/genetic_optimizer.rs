use crate::evaluators::{GeneticEvaluator, GeneticEvaluatorFactory};
use crate::fitness_calculator::FitnessCalculator;
use rand::Rng;
use rust_reversi_core::search::Evaluator;
use std::time::Duration;

pub struct GeneticRateConfig {
    mutation_rate: f64,
    crossover_rate: f64,
    selection_rate: f64,
}

impl GeneticRateConfig {
    pub fn new(mutation_rate: f64, crossover_rate: f64, selection_rate: f64) -> GeneticRateConfig {
        GeneticRateConfig {
            mutation_rate,
            crossover_rate,
            selection_rate,
        }
    }
}

pub struct GeneticOptimizer<Y: GeneticEvaluatorFactory, Z: FitnessCalculator> {
    population: Vec<Box<dyn GeneticEvaluator>>,
    population_size: usize,
    generation: usize,
    rate_config: GeneticRateConfig,
    tournament_size: usize,
    timeout: Duration,
    epsilon: f64,
    factory: Y,
    fitness_calculator: Z,
}

impl<Y: GeneticEvaluatorFactory, Z: FitnessCalculator> GeneticOptimizer<Y, Z> {
    fn initialize_population(size: usize, factory: &Y) -> Vec<Box<dyn GeneticEvaluator>> {
        let mut population = Vec::new();
        for _ in 0..size {
            population.push(factory.generate());
        }
        population
    }

    pub fn new(
        population_size: usize,
        rate_config: GeneticRateConfig,
        tournament_size: usize,
        timeout: Duration,
        epsilon: f64,
        factory: Y,
        fitness_calculator: Z,
    ) -> GeneticOptimizer<Y, Z> {
        GeneticOptimizer {
            population: GeneticOptimizer::<Y, Z>::initialize_population(population_size, &factory),
            population_size,
            generation: 0,
            rate_config,
            tournament_size,
            timeout,
            epsilon,
            factory,
            fitness_calculator,
        }
    }

    fn evaluate_fitness(&self) -> Vec<f64> {
        let mut fitnesses = Vec::new();
        for individual in &self.population {
            fitnesses.push(
                self.fitness_calculator
                    .calculate_fitness(individual.to_evaluator()),
            );
        }
        fitnesses
    }

    fn select_parent(&self, fitnesses: &Vec<f64>) -> &Box<dyn GeneticEvaluator> {
        let mut rng = rand::thread_rng();
        let mut best = rng.gen_range(0..self.population_size);
        for _ in 0..self.tournament_size {
            let candidate = rng.gen_range(0..self.population_size);
            if fitnesses[candidate] > fitnesses[best] {
                best = candidate;
            }
        }
        &self.population[best]
    }

    fn evolve(&mut self) {
        let fitnesses = self.evaluate_fitness();
        let mut new_population = Vec::new();
        for _ in 0..self.population_size {
            let parent1 = self.select_parent(&fitnesses);
            let parent2 = self.select_parent(&fitnesses);
            let mut rng = rand::thread_rng();
            let child = if rng.gen_bool(self.rate_config.crossover_rate) {
                parent1.crossover(&**parent2)
            } else {
                parent1.mutate()
            };
            new_population.push(child);
        }
        self.population = new_population;
        self.generation += 1;
    }

    pub fn optimize(&self) -> Box<dyn Evaluator> {
        unimplemented!()
    }
}
