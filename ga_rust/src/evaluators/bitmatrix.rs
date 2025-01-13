use crate::evaluators::{GeneticEvaluator, GeneticEvaluatorFactory};
use rand::Rng;
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

impl<const N: usize> std::fmt::Debug for GeneticBitMatrixEvaluator<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result = String::new();
        result.push_str("GeneticBitMatrixEvaluator {\n");
        result.push_str("\tmasks: [\n");
        for mask in self.masks.iter() {
            result.push_str(&format!("\t\t{:x},\n", mask));
        }
        result.push_str("\t],\n");
        result.push_str("\tweights: [\n");
        for weight in self.weights.iter() {
            result.push_str(&format!("\t\t{},\n", weight));
        }
        result.push_str("\t],\n}");
        write!(f, "{}", result)
    }
}

impl<const N: usize> GeneticEvaluator for GeneticBitMatrixEvaluator<N> {
    fn mutate(&self) -> Box<dyn GeneticEvaluator> {
        let mut rng = rand::thread_rng();
        let mut masks = self.masks;
        let mut weights = self.weights;
        let index = rng.gen_range(0..N);
        let mut mask = 0;
        for _ in 0..64 {
            if rng.gen_bool(0.2) {
                mask |= 1;
            }
            mask <<= 1;
        }
        masks[index] = mask;
        weights[index] = rng.gen_range(-10..10);
        Box::new(GeneticBitMatrixEvaluator::<N>::new(
            masks.to_vec(),
            weights.to_vec(),
        ))
    }

    fn crossover(&self, other: &dyn GeneticEvaluator) -> Box<dyn GeneticEvaluator> {
        Box::new(GeneticBitMatrixEvaluator::<N>::new(
            self.masks.to_vec(),
            self.weights.to_vec(),
        ))
    }

    fn to_evaluator(&self) -> Box<dyn Evaluator> {
        let evaluator = BitMatrixEvaluator::<N>::new(self.weights.to_vec(), self.masks.to_vec());
        Box::new(evaluator)
    }
}

pub struct GeneticBitMatrixEvaluatorFactory<const N: usize> {}

impl<const N: usize> GeneticBitMatrixEvaluatorFactory<N> {
    pub fn new() -> GeneticBitMatrixEvaluatorFactory<N> {
        GeneticBitMatrixEvaluatorFactory {}
    }
}

impl<const N: usize> GeneticEvaluatorFactory for GeneticBitMatrixEvaluatorFactory<N> {
    fn generate(&self) -> Box<dyn GeneticEvaluator> {
        let mut rng = rand::thread_rng();
        let mut masks = Vec::new();
        let mut weights = Vec::new();
        for _ in 0..N {
            // masks.push(rng.gen());
            let mut mask = 0;
            for _ in 0..64 {
                if rng.gen_bool(0.2) {
                    mask |= 1;
                }
                mask <<= 1;
            }
            masks.push(mask);
            weights.push(rng.gen_range(-10..10));
        }
        Box::new(GeneticBitMatrixEvaluator::<N>::new(masks, weights))
    }
}
