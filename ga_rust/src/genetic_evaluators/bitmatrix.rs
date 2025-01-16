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
        let masks = [
            0x0000001818000000,
            0x0000182424180000,
            0x0000240000240000,
            0x0018004242001800,
            0x0024420000422400,
            0x0042000000004200,
            0x1800008181000018,
            0x2400810000810024,
            0x4281000000008142,
            0x8100000000000081,
        ];
        let mut weights = [0; N];
        for w in weights.iter_mut() {
            *w = rng.gen_range(-10..10);
        }
        GeneticBitMatrixEvaluator::<N>::new(masks.to_vec(), weights.to_vec())
    }

    pub fn mutate(&self) -> GeneticBitMatrixEvaluator<N> {
        let mut rng = rand::thread_rng();
        let masks = self.masks;
        let mut weights = self.weights;
        for w in weights.iter_mut() {
            if rng.gen_bool(0.5) {
                *w += rng.gen_range(-5..5);
            }
        }
        GeneticBitMatrixEvaluator::<N>::new(masks.to_vec(), weights.to_vec())
    }

    pub fn crossover(&self, other: &GeneticBitMatrixEvaluator<N>) -> GeneticBitMatrixEvaluator<N> {
        let mut rng = rand::thread_rng();
        let masks = self.masks;
        let mut weights = self.weights;
        for (i, w) in weights.iter_mut().enumerate() {
            if rng.gen_bool(0.5) {
                *w = other.weights[i];
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
