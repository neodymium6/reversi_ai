use crate::{
    evaluators, fitness_calculator::bitmatrix::EvaluatorType,
    genetic_evaluators::bitmatrix::GeneticBitMatrixEvaluator,
};
use evaluators::multi_bitmatrix::MultiBitMatrixEvaluator;
use rand::Rng;

use super::GeneticEvaluator;

#[derive(Clone)]
pub struct GeneticMultiBitMatrixEvaluator<const N: usize> {
    evaluators: Vec<GeneticBitMatrixEvaluator<N>>,
    bounds: Vec<usize>,
}

impl<const N: usize> GeneticMultiBitMatrixEvaluator<N> {
    pub fn new(
        evaluators: Vec<GeneticBitMatrixEvaluator<N>>,
        bounds: Vec<usize>,
    ) -> GeneticMultiBitMatrixEvaluator<N> {
        assert_eq!(evaluators.len(), bounds.len() + 1);
        let mut current_bound = 0;
        for bound in bounds.iter() {
            assert!(current_bound < *bound && *bound < 64);
            current_bound = *bound;
        }
        GeneticMultiBitMatrixEvaluator { evaluators, bounds }
    }
}

impl<const N: usize> GeneticEvaluator<N> for GeneticMultiBitMatrixEvaluator<N> {
    fn mutate(&self) -> GeneticMultiBitMatrixEvaluator<N> {
        let mut rng = rand::thread_rng();
        let new_evaluators = self
            .evaluators
            .iter()
            .map(|evaluator| match rng.gen_bool(0.5) {
                true => evaluator.mutate(),
                false => evaluator.clone(),
            })
            .collect();
        GeneticMultiBitMatrixEvaluator::<N>::new(new_evaluators, self.bounds.clone())
    }

    fn crossover(
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

    fn new_from_random() -> Self {
        let bounds = vec![20, 35, 50];
        let mut evaluators = Vec::new();
        for _i in 0..bounds.len() + 1 {
            evaluators.push(GeneticBitMatrixEvaluator::<N>::new_from_random());
        }
        GeneticMultiBitMatrixEvaluator::<N>::new(evaluators, bounds)
    }

    fn to_evaluator_type(&self) -> EvaluatorType<N> {
        let evaluators = self
            .evaluators
            .iter()
            .map(|evaluator| evaluator.to_bitmatrix_evaluator())
            .collect();
        EvaluatorType::MultiBitMatrix(Box::new(MultiBitMatrixEvaluator::<N>::new(
            evaluators,
            self.bounds.clone(),
        )))
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
