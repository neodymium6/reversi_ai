use crate::evaluators::{GeneticEvaluator, GeneticEvaluatorFactory};
use crate::fitness_calculator::FitnessCalculator;
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

pub struct GeneticOptimizer<
    X: GeneticEvaluator,
    Y: GeneticEvaluatorFactory<X>,
    Z: FitnessCalculator<X>,
> {
    population: Vec<Box<X>>,
    population_size: usize,
    generation: usize,
    rate_config: GeneticRateConfig,
    tournament_size: usize,
    timeout: Duration,
    epsilon: f64,
    factory: Y,
    fitness_calculator: Z,
}

impl<X: GeneticEvaluator, Y: GeneticEvaluatorFactory<X>, Z: FitnessCalculator<X>>
    GeneticOptimizer<X, Y, Z>
{
    fn initialize_population(size: usize, factory: &Y) -> Vec<Box<X>> {
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
    ) -> GeneticOptimizer<X, Y, Z> {
        GeneticOptimizer {
            population: GeneticOptimizer::<X, Y, Z>::initialize_population(
                population_size,
                &factory,
            ),
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
            fitnesses.push(self.fitness_calculator.calculate_fitness(individual));
        }
        fitnesses
    }

    pub fn optimize(&self) -> Box<dyn Evaluator> {
        unimplemented!()
    }
}
