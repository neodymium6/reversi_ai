pub mod simple;

use rust_reversi_core::search::Evaluator;

pub trait FitnessCalculator {
    fn calculate_fitness(&self, evaluator: Box<dyn Evaluator>) -> f64;
}
