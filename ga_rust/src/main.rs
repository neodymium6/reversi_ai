mod evaluator_evaluator;
mod evaluators;
mod fitness_calculator;
mod genetic_optimizer;
use fitness_calculator::bitmatrix::{EvaluatorType, MultiFitnessCalculator};
use genetic_optimizer::OptimizerConfig;
use rust_reversi_core::search::BitMatrixEvaluator;
fn main() {
    let masks: Vec<u64> = vec![
        0x0000240000240000,
        0x0024420000422400,
        0x0042000000004200,
        0x1800008181000018,
        0x2400810000810024,
        0x4281000000008142,
        0x8100000000000081,
        0,
        0,
        0,
    ];
    let weights1: Vec<i32> = vec![1, -1, -7, 3, -1, -3, 15, 0, 0, 0];
    let weights2: Vec<i32> = vec![-5, 0, -5, 3, -5, -10, 0, 4, 7, 9];
    let evaluator_vec: Vec<EvaluatorType> = vec![
        EvaluatorType::BitMatrix(BitMatrixEvaluator::<10>::new(weights1, masks.clone())),
        EvaluatorType::BitMatrix(BitMatrixEvaluator::<10>::new(weights2, masks)),
        EvaluatorType::Piece,
        EvaluatorType::LegalNum,
    ];
    let fitness_calculator = MultiFitnessCalculator::<10>::new(evaluator_vec);
    let config = OptimizerConfig {
        population_size: 100,
        mutation_rate: 0.2,
        crossover_rate: 0.5,
        tournament_size: 5,
        max_generations: 10,
    };
    let mut optimizer =
        genetic_optimizer::bitmatrix::BitMatrixOptimizer::new(Box::new(fitness_calculator), config);
    optimizer.optimize();
}
