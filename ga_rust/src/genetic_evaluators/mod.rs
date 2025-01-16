pub mod bitmatrix;
pub mod multi_bitmatrix;
use rust_reversi_core::search::Evaluator;

pub trait GeneticEvaluator {
    fn new_from_random() -> Self;
    fn mutate(&self) -> Self;
    fn crossover(&self, other: &Self) -> Self;
    fn to_evaluator(&self) -> Box<dyn Evaluator>;
}
