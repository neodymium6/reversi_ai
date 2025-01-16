use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use crate::evaluator_evaluator::EvaluatorEvaluator;

use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use rust_reversi_core::search::{
    BitMatrixEvaluator, Evaluator, LegalNumEvaluator, MatrixEvaluator, PieceEvaluator,
};

#[derive(Clone)]
pub enum EvaluatorType {
    Piece,
    LegalNum,
    Matrix(MatrixEvaluator),
    BitMatrix(BitMatrixEvaluator<10>),
}

impl EvaluatorType {
    pub fn to_evaluator(&self) -> Box<dyn Evaluator> {
        match self {
            EvaluatorType::Piece => Box::new(PieceEvaluator::new()),
            EvaluatorType::LegalNum => Box::new(LegalNumEvaluator::new()),
            EvaluatorType::Matrix(evaluator) => Box::new(evaluator.clone()),
            EvaluatorType::BitMatrix(evaluator) => Box::new(evaluator.clone()),
        }
    }
}

pub trait FitnessCalculator<const N: usize> {
    fn calculate_fitness(&self, evaluators: Vec<BitMatrixEvaluator<N>>) -> Vec<f64>;
}

#[derive(Clone)]
pub struct SimpleFitnessCalculator<const N: usize> {
    evaluator: EvaluatorType,
}

impl<const N: usize> SimpleFitnessCalculator<N> {
    pub fn new(evaluator: EvaluatorType) -> SimpleFitnessCalculator<N> {
        SimpleFitnessCalculator { evaluator }
    }

    fn vs_self(&self, evaluator: BitMatrixEvaluator<N>) -> f64 {
        let evaluator_evaluator = EvaluatorEvaluator::new(
            self.evaluator.to_evaluator(),
            Box::new(evaluator),
            std::time::Duration::from_millis(10),
            2,
            1,
            0.1,
            false,
        );
        let (_self_score, arg_score) = evaluator_evaluator.eval_with_depth(100);
        arg_score
    }
}

impl<const N: usize> FitnessCalculator<N> for SimpleFitnessCalculator<N> {
    fn calculate_fitness(&self, evaluators: Vec<BitMatrixEvaluator<N>>) -> Vec<f64> {
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

pub struct MultiFitnessCalculator<const N: usize> {
    fitness_calculators: Vec<SimpleFitnessCalculator<N>>,
    weights: Vec<f64>,
}

impl<const N: usize> MultiFitnessCalculator<N> {
    pub fn new(evaluators: Vec<(EvaluatorType, f64)>) -> MultiFitnessCalculator<N> {
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
    fn calculate_fitness(&self, evaluators: Vec<BitMatrixEvaluator<N>>) -> Vec<f64> {
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
                    .map(|fitness_calculator| fitness_calculator.vs_self(evaluator.clone()))
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
