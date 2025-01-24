use std::sync::{Arc, Mutex};

use crate::evaluator_evaluator::EvaluatorEvaluator;

use crate::evaluators::multi_bitmatrix::MultiBitMatrixEvaluator;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use rayon::prelude::*;
use rust_reversi_core::search::{BitMatrixEvaluator, Evaluator};

#[derive(Clone, Debug)]
pub enum EvaluatorType<const N: usize> {
    // Piece,
    // LegalNum,
    // Matrix(MatrixEvaluator),
    BitMatrix(Box<BitMatrixEvaluator<N>>),
    MultiBitMatrix(Box<MultiBitMatrixEvaluator<N>>),
}

impl<const N: usize> EvaluatorType<N> {
    pub fn to_evaluator(&self) -> Box<dyn Evaluator> {
        match self {
            // EvaluatorType::Piece => Box::new(PieceEvaluator::new()),
            // EvaluatorType::LegalNum => Box::new(LegalNumEvaluator::new()),
            // EvaluatorType::Matrix(evaluator) => Box::new(evaluator.clone()),
            EvaluatorType::BitMatrix(evaluator) => evaluator.clone(),
            EvaluatorType::MultiBitMatrix(evaluator) => evaluator.clone(),
        }
    }
}

impl<const N: usize> Default for EvaluatorType<N> {
    fn default() -> Self {
        EvaluatorType::BitMatrix(Box::new(BitMatrixEvaluator::<N>::new(
            vec![0; N],
            vec![0; N],
        )))
    }
}

pub trait FitnessCalculator<const N: usize> {
    fn calculate_fitness(&self, evaluators: Vec<EvaluatorType<N>>) -> Vec<f64>;
}

#[derive(Clone)]
pub struct SimpleFitnessCalculator<const N: usize> {
    evaluator: EvaluatorType<N>,
}

impl<const N: usize> SimpleFitnessCalculator<N> {
    pub fn new(evaluator: EvaluatorType<N>) -> SimpleFitnessCalculator<N> {
        SimpleFitnessCalculator { evaluator }
    }

    fn vs_self(&self, evaluator: Box<dyn Evaluator>) -> f64 {
        let evaluator_evaluator = EvaluatorEvaluator::new(
            self.evaluator.to_evaluator(),
            evaluator,
            std::time::Duration::from_millis(10),
            2,
            1,
            0.1,
            false,
        );
        let (_self_score, arg_score) = evaluator_evaluator.eval_with_depth(20);
        arg_score
    }
}

impl<const N: usize> FitnessCalculator<N> for SimpleFitnessCalculator<N> {
    fn calculate_fitness(&self, evaluators: Vec<EvaluatorType<N>>) -> Vec<f64> {
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
                let fitness = self.vs_self(evaluator.to_evaluator());
                let pb = pb.lock().unwrap();
                pb.inc(1);
                fitness
            })
            .collect()
    }
}

pub struct MultiFitnessCalculator<const N: usize> {
    fitness_calculators: Vec<SimpleFitnessCalculator<N>>,
    weights: Vec<f64>,
}

impl<const N: usize> MultiFitnessCalculator<N> {
    pub fn new(evaluators: Vec<(EvaluatorType<N>, f64)>) -> MultiFitnessCalculator<N> {
        let fitness_calculators = evaluators
            .iter()
            .map(|evaluator| SimpleFitnessCalculator::<N>::new(evaluator.0.clone()))
            .collect();
        let weights = evaluators.iter().map(|evaluator| evaluator.1).collect();
        MultiFitnessCalculator {
            fitness_calculators,
            weights,
        }
    }
}

impl<const N: usize> FitnessCalculator<N> for MultiFitnessCalculator<N> {
    fn calculate_fitness(&self, evaluators: Vec<EvaluatorType<N>>) -> Vec<f64> {
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
                let fitness = self
                    .fitness_calculators
                    .iter()
                    .map(|fitness_calculator| fitness_calculator.vs_self(evaluator.to_evaluator()))
                    .zip(self.weights.iter())
                    .map(|(fitness, weight)| fitness * weight)
                    .sum();
                let pb = pb.lock().unwrap();
                pb.inc(1);
                fitness
            })
            .collect()
    }
}
