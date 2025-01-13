mod evaluator_evaluator;
mod evaluators;
mod fitness_calculator;
mod genetic_optimizer;
use evaluators::bitmatrix::GeneticBitMatrixEvaluatorFactory;
use fitness_calculator::simple::SimpleFitnessCalculator;
use genetic_optimizer::GeneticOptimizer;
use genetic_optimizer::GeneticRateConfig;
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
    let weights: Vec<i32> = vec![1, -1, -7, 3, -1, -3, 15, 0, 0, 0];
    let evaluator = BitMatrixEvaluator::<10>::new(weights, masks);
    let rate_config = GeneticRateConfig::new(0.1, 0.1, 0.1);
    let mut optimizer = GeneticOptimizer::new(
        10,
        rate_config,
        5,
        10,
        GeneticBitMatrixEvaluatorFactory::<10>::new(),
        SimpleFitnessCalculator::new(evaluator),
    );
    optimizer.optimize();
}
