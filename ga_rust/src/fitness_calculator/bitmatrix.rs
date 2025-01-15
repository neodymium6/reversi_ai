use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use crate::evaluator_evaluator::EvaluatorEvaluator;

use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use rust_reversi_core::search::BitMatrixEvaluator;

#[derive(Clone)]
pub struct SimpleFitnessCalculator<const N: usize> {
    evaluator: BitMatrixEvaluator<10>,
}

impl<const N: usize> SimpleFitnessCalculator<N> {
    pub fn new(evaluator: BitMatrixEvaluator<10>) -> SimpleFitnessCalculator<10> {
        SimpleFitnessCalculator { evaluator }
    }

    fn vs_self(&self, evaluator: BitMatrixEvaluator<N>) -> f64 {
        let evaluator_evaluator = EvaluatorEvaluator::new(
            Box::new(self.evaluator.clone()),
            Box::new(evaluator),
            std::time::Duration::from_millis(10),
            3,
            3,
            0.1,
            false,
        );
        // let (_self_score, arg_score) = evaluator_evaluator.eval(100);
        let (_self_score, arg_score) = evaluator_evaluator.eval_with_depth(100);
        arg_score
    }

    pub fn calculate_fitness(&self, evaluators: Vec<BitMatrixEvaluator<N>>) -> Vec<f64> {
        let pb = ProgressBar::new(evaluators.len() as u64);
        pb.set_style(
            ProgressStyle::with_template("[{wide_bar}] [{elapsed_precise}] ({eta})")
                .unwrap()
                .with_key(
                    "eta",
                    |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                        write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                    },
                )
                .progress_chars("#>-"),
        );
        let pb = Arc::new(Mutex::new(pb));
        evaluators
            .par_iter()
            .map(|evaluator| {
                let fitness = self.vs_self(evaluator.clone());
                let pb = pb.lock().unwrap();
                pb.inc(1);
                fitness
            })
            .collect()
    }
}
