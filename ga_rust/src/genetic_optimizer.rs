use crate::evaluator_evaluator::EvaluatorEvaluator;
use crate::evaluators::{GeneticEvaluator, GeneticEvaluatorFactory};
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

pub struct GeneticOptimizer<X: GeneticEvaluator, Y: GeneticEvaluatorFactory<X>> {
    population: Vec<Box<X>>,
    population_size: usize,
    generation: usize,
    rate_config: GeneticRateConfig,
    tournament_size: usize,
    timeout: Duration,
    epsilon: f64,
    factory: Y,
}

impl<X: GeneticEvaluator, Y: GeneticEvaluatorFactory<X>> GeneticOptimizer<X, Y> {
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
    ) -> GeneticOptimizer<X, Y> {
        GeneticOptimizer {
            population: GeneticOptimizer::initialize_population(population_size, &factory),
            population_size,
            generation: 0,
            rate_config,
            tournament_size,
            timeout,
            epsilon,
            factory,
        }
    }

    fn evaluate_fitness(&self) -> Vec<f64> {
        unimplemented!()
    }

    pub fn optimize(&self) -> Box<dyn Evaluator> {
        unimplemented!()
    }
}
