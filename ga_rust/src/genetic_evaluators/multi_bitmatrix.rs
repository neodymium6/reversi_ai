use rand::Rng;

use crate::{evaluators, genetic_evaluators::bitmatrix::GeneticBitMatrixEvaluator};
use evaluators::multi_bitmatrix::MultiBitMatrixEvaluator;

pub struct GeneticMultiBitMatrixEvaluator<const N: usize> {
    evaluators: Vec<GeneticBitMatrixEvaluator<N>>,
    bounds: Vec<usize>,
}

impl<const N: usize> GeneticMultiBitMatrixEvaluator<N> {
    pub fn new(
        evaluators: Vec<GeneticBitMatrixEvaluator<N>>,
        bounds: Vec<usize>,
    ) -> GeneticMultiBitMatrixEvaluator<N> {
        assert_eq!(evaluators.len(), bounds.len() - 1);
        let mut current_bound = 0;
        for bound in bounds.iter() {
            assert!(current_bound < *bound && *bound < 64);
            current_bound = *bound;
        }
        GeneticMultiBitMatrixEvaluator { evaluators, bounds }
    }

    pub fn new_from_random(bounds: Vec<usize>) {
        let mut evaluators = Vec::new();
        for i in 0..bounds.len() - 1 {
            evaluators.push(GeneticBitMatrixEvaluator::<N>::new_from_random());
        }
    }

    pub fn mutate(&self) -> GeneticMultiBitMatrixEvaluator<N> {
        let mut rng = rand::thread_rng();
        let new_evaluators = self
            .evaluators
            .iter()
            .map(|evaluator| {
                if rng.gen_bool(0.5) {
                    evaluator.mutate()
                } else {
                    evaluator.clone()
                }
            })
            .collect();
        GeneticMultiBitMatrixEvaluator::<N>::new(new_evaluators, self.bounds.clone())
    }

    pub fn crossover(
        &self,
        other: &GeneticMultiBitMatrixEvaluator<N>,
    ) -> GeneticMultiBitMatrixEvaluator<N> {
        let new_evaluators = self
            .evaluators
            .iter()
            .zip(other.evaluators.iter())
            .map(|(evaluator, other_evaluator)| evaluator.crossover(other_evaluator))
            .collect();
        GeneticMultiBitMatrixEvaluator::<N>::new(new_evaluators, self.bounds.clone())
    }

    pub fn to_evaluator(&self) -> MultiBitMatrixEvaluator<N> {
        let evaluators = self
            .evaluators
            .iter()
            .map(|evaluator| evaluator.to_evaluator())
            .collect();
        MultiBitMatrixEvaluator::<N>::new(evaluators, self.bounds.clone())
    }
}

impl<const N: usize> std::fmt::Debug for GeneticMultiBitMatrixEvaluator<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GeneticMultiBitMatrixEvaluator")?;
        for (i, evaluator) in self.evaluators.iter().enumerate() {
            writeln!(f, "evaluator:{}", i)?;
            writeln!(f, "{:?}", evaluator)?;
        }
        Ok(())
    }
}
