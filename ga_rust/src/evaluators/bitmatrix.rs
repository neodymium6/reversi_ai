use crate::evaluators::{GeneticEvaluator, GeneticEvaluatorFactory};
use rust_reversi_core::search::BitMatrixEvaluator;
use rust_reversi_core::search::Evaluator;

pub struct GeneticBitMatrixEvaluator<const N: usize> {
    evaluator: BitMatrixEvaluator<N>,
    masks: [u64; N],
    weights: [i32; N],
}

impl<const N: usize> GeneticBitMatrixEvaluator<N> {
    pub fn new(masks: Vec<u64>, weights: Vec<i32>) -> GeneticBitMatrixEvaluator<N> {
        let mut masks_arr = [0; N];
        let mut weights_arr = [0; N];
        masks_arr.copy_from_slice(&masks);
        weights_arr.copy_from_slice(&weights);
        let evaluator = BitMatrixEvaluator::<N>::new(weights, masks);
        GeneticBitMatrixEvaluator {
            evaluator,
            masks: masks_arr,
            weights: weights_arr,
        }
    }
}

impl<const N: usize> Evaluator for GeneticBitMatrixEvaluator<N> {
    fn evaluate(&self, board: &rust_reversi_core::board::Board) -> i32 {
        self.evaluator.evaluate(board)
    }
}

impl<const N: usize> GeneticEvaluator for GeneticBitMatrixEvaluator<N> {
    fn mutate(&self) -> Box<dyn GeneticEvaluator> {
        unimplemented!()
    }

    fn crossover(&self, other: &dyn GeneticEvaluator) -> Box<dyn GeneticEvaluator> {
        unimplemented!()
    }
}

pub struct GeneticBitMatrixEvaluatorFactory<const N: usize> {}

impl<const N: usize> GeneticEvaluatorFactory<GeneticBitMatrixEvaluator<N>>
    for GeneticBitMatrixEvaluatorFactory<N>
{
    fn generate(&self) -> Box<GeneticBitMatrixEvaluator<N>> {
        unimplemented!()
    }
}
