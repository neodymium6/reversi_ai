mod bitmatrix;

use rust_reversi_core::search::Evaluator;

pub trait GeneticEvaluator: Evaluator {
    fn mutate(&self) -> Box<dyn GeneticEvaluator>;
    fn crossover(&self, other: &dyn GeneticEvaluator) -> Box<dyn GeneticEvaluator>;
}

pub trait GeneticEvaluatorFactory<X: GeneticEvaluator> {
    fn generate(&self) -> Box<X>;
}
