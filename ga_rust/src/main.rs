mod evaluator_evaluator;
mod evaluators;
mod fitness_calculator;
mod genetic_optimizer;
use fitness_calculator::bitmatrix::{EvaluatorType, MultiFitnessCalculator};
use genetic_optimizer::OptimizerConfig;
use rust_reversi_core::search::BitMatrixEvaluator;
fn main() {
    let masks: Vec<u64> = vec![
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
    let evaluators = vec![
        // あなたの最強の重み
        vec![0, 0, 1, 0, -1, -7, 3, -1, -3, 15],
        // 中央重視系
        vec![4, 3, -1, -2, -3, -5, 2, -2, -4, 12],
        vec![5, 4, 2, -3, -4, -8, 1, -3, -5, 14],
        // 外周重視系
        vec![-2, -1, 2, 3, 4, -6, 5, 4, -2, 16],
        vec![-3, -2, 1, 4, 5, -7, 6, 5, -3, 18],
        // より極端な重み付け
        vec![8, -6, -4, -2, 3, -12, 7, -6, -8, 22],
        vec![-7, 5, 6, -4, -5, -14, 4, 3, -7, 24],
        // 控えめな重み
        vec![1, 1, 0, 1, -2, -4, 2, -1, -2, 8],
        vec![0, 2, 1, -1, -2, -5, 1, -2, -3, 10],
        // 実験的な重み
        vec![4, -3, 5, -4, 6, -10, -3, 4, -6, 21],
    ];

    let weights = [
        0.15, // 最強の重み
        0.12, // 中央重視1
        0.12, // 中央重視2
        0.10, // 外周重視1
        0.10, // 外周重視2
        0.08, // 極端1
        0.08, // 極端2
        0.09, // 控えめ1
        0.09, // 控えめ2
        0.07,
    ];
    let evaluator_vec: Vec<EvaluatorType> = evaluators
        .into_iter()
        .map(|weights| {
            EvaluatorType::BitMatrix(BitMatrixEvaluator::<10>::new(weights, masks.clone()))
        })
        .collect();
    let fitness_calculator = MultiFitnessCalculator::<10>::new(
        evaluator_vec
            .iter()
            .enumerate()
            .map(|(i, evaluator)| (evaluator.clone(), weights[i]))
            .collect(),
    );
    let config = OptimizerConfig {
        population_size: 100,
        mutation_rate: 0.2,
        crossover_rate: 0.5,
        tournament_size: 5,
        max_generations: 100,
    };
    let mut optimizer =
        genetic_optimizer::bitmatrix::BitMatrixOptimizer::new(Box::new(fitness_calculator), config);
    optimizer.optimize();
}
