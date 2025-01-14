use rand::Rng;
use rust_reversi_core::search::BitMatrixEvaluator;

#[derive(Clone)]
pub struct GeneticBitMatrixEvaluator<const N: usize> {
    masks: [u64; N],
    weights: [i32; N],
}

impl<const N: usize> GeneticBitMatrixEvaluator<N> {
    pub fn new(masks: Vec<u64>, weights: Vec<i32>) -> GeneticBitMatrixEvaluator<N> {
        let mut masks_arr = [0; N];
        let mut weights_arr = [0; N];
        masks_arr.copy_from_slice(&masks);
        weights_arr.copy_from_slice(&weights);
        GeneticBitMatrixEvaluator {
            masks: masks_arr,
            weights: weights_arr,
        }
    }

    pub fn new_from_random() -> GeneticBitMatrixEvaluator<N> {
        let mut rng = rand::thread_rng();
        let mut masks = [0; N];
        let mut weights = [0; N];
        for i in 0..N {
            for _ in 0..64 {
                if rng.gen_bool(0.2) {
                    masks[i] |= 1;
                }
                masks[i] <<= 1;
            }
            weights[i] = rng.gen_range(-10..10);
        }
        GeneticBitMatrixEvaluator::<N>::new(masks.to_vec(), weights.to_vec())
    }

    pub fn mutate(&self) -> GeneticBitMatrixEvaluator<N> {
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
        GeneticBitMatrixEvaluator::<N>::new(masks.to_vec(), weights.to_vec())
    }

    pub fn crossover(&self, other: &GeneticBitMatrixEvaluator<N>) -> GeneticBitMatrixEvaluator<N> {
        let mut rng = rand::thread_rng();
        let mut masks = self.masks;
        let mut weights = self.weights;
        for i in 0..N {
            if rng.gen_bool(0.5) {
                // masks[i] = self.masks[i] ^ other.masks[i];
                masks[i] = match rng.gen_bool(0.5) {
                    true => self.masks[i] ^ other.masks[i],
                    false => self.masks[i] | other.masks[i],
                };
                weights[i] = (self.weights[i] + other.weights[i]) / 2;
            } else if rng.gen_bool(0.5) {
                masks[i] = other.masks[i];
                weights[i] = other.weights[i];
            }
        }
        GeneticBitMatrixEvaluator::<N>::new(masks.to_vec(), weights.to_vec())
    }

    pub fn to_evaluator(&self) -> BitMatrixEvaluator<N> {
        BitMatrixEvaluator::<N>::new(self.weights.to_vec(), self.masks.to_vec())
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
