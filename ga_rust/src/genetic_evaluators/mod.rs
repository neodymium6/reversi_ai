pub mod bitmatrix;
pub mod multi_bitmatrix;
use std::fmt::Debug;

use rust_reversi_core::search::Evaluator;

use crate::fitness_calculator::bitmatrix::EvaluatorType;

pub trait GeneticEvaluator<const N: usize>: Clone + Debug {
    fn new_from_random() -> Self;
    fn mutate(&self) -> Self;
    fn crossover(&self, other: &Self) -> Self;
    fn to_evaluator(&self) -> Box<dyn Evaluator>;
    fn to_evaluator_type(&self) -> EvaluatorType<N>;
}
