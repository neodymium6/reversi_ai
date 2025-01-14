use crate::evaluator_evaluator::EvaluatorEvaluator;

use rust_reversi_core::search::BitMatrixEvaluator;

#[derive(Clone)]
pub struct SimpleFitnessCalculator<const N: usize> {
    evaluator: BitMatrixEvaluator<10>,
}

impl<const N: usize> SimpleFitnessCalculator<N> {
    pub fn new(evaluator: BitMatrixEvaluator<10>) -> SimpleFitnessCalculator<10> {
        SimpleFitnessCalculator { evaluator }
    }

    pub fn calculate_fitness(&self, evaluator: BitMatrixEvaluator<N>) -> f64 {
        let evaluator_evaluator = EvaluatorEvaluator::new(
            Box::new(self.evaluator.clone()),
            Box::new(evaluator),
            std::time::Duration::from_millis(10),
            3,
            0.1,
            false,
        );
        // let (_self_score, arg_score) = evaluator_evaluator.eval(100);
        let (_self_score, arg_score) = evaluator_evaluator.eval_with_depth(100);
        arg_score
    }
}
