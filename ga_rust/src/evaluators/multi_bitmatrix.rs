use rust_reversi_core::{
    board::Board,
    search::{BitMatrixEvaluator, Evaluator},
};

#[derive(Clone)]
pub struct MultiBitMatrixEvaluator<const N: usize> {
    evaluators: Vec<BitMatrixEvaluator<N>>,
    evaluator_mapping: [usize; 65],
}

impl<const N: usize> MultiBitMatrixEvaluator<N> {
    pub fn new(
        evaluators: Vec<BitMatrixEvaluator<N>>,
        bounds: Vec<usize>,
    ) -> MultiBitMatrixEvaluator<N> {
        assert_eq!(evaluators.len(), bounds.len() + 1);
        let mut current_bound = 0;
        let mut evaluator_mapping = [0; 65];
        for (i, bound) in bounds.iter().enumerate() {
            assert!(current_bound < *bound && *bound < 64);
            evaluator_mapping[current_bound..*bound].fill(i);
            current_bound = *bound;
        }
        evaluator_mapping[current_bound..65].fill(evaluators.len() - 1);
        MultiBitMatrixEvaluator {
            evaluators,
            evaluator_mapping,
        }
    }
}

impl<const N: usize> Evaluator for MultiBitMatrixEvaluator<N> {
    fn evaluate(&self, board: &Board) -> i32 {
        let piece_sum = board.piece_sum() as usize;
        let evaluator = &self.evaluators[self.evaluator_mapping[piece_sum]];
        evaluator.evaluate(board)
    }
}
