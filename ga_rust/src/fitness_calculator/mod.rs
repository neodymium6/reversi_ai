use crate::evaluators::GeneticEvaluator;

pub trait FitnessCalculator<X: GeneticEvaluator> {
    fn calculate_fitness(&self, evaluator: &Box<X>) -> f64;
}
