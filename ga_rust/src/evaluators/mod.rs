mod bitmatrix;

use rust_reversi_core::search::Evaluator;

pub trait GeneticEvaluator: Evaluator {
    fn mutate(&self) -> Box<dyn GeneticEvaluator>;
    fn crossover(&self, other: &dyn GeneticEvaluator) -> Box<dyn GeneticEvaluator>;
    fn to_evaluator(&self) -> Box<dyn Evaluator>;
}

pub trait GeneticEvaluatorFactory {
    fn generate(&self) -> Box<dyn GeneticEvaluator>;
}
