use crate::evaluator_evaluator::EvaluatorEvaluator;
use crate::fitness_calculator::FitnessCalculator;
use rust_reversi_core::search::{BitMatrixEvaluator, Evaluator};

pub struct SimpleFitnessCalculator {
    evaluator: BitMatrixEvaluator<10>,
}

impl SimpleFitnessCalculator {
    pub fn new(evaluator: BitMatrixEvaluator<10>) -> SimpleFitnessCalculator {
        SimpleFitnessCalculator { evaluator }
    }
}

impl FitnessCalculator for SimpleFitnessCalculator {
    fn calculate_fitness(&self, evaluator: Box<dyn Evaluator>) -> f64 {
        let evaluator_evaluator = EvaluatorEvaluator::new(
            Box::new(self.evaluator.clone()),
            evaluator,
            std::time::Duration::from_millis(2),
            0.1,
        );
        let (self_score, arg_score) = evaluator_evaluator.eval(100);
        arg_score
    }
}
