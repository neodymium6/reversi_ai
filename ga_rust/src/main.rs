mod evaluator_evaluator;
mod evaluator_pool;
mod evaluators;
mod fitness_calculator;
mod genetic_evaluators;
mod genetic_optimizer;
use evaluator_pool::RankedEvaluatorPool;
use fitness_calculator::bitmatrix::{EvaluatorType, MultiFitnessCalculator};
use genetic_evaluators::multi_bitmatrix::GeneticMultiBitMatrixEvaluator;
use genetic_evaluators::GeneticEvaluator;
use genetic_optimizer::GeneticOptimizer;
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
        // === コアパターングループ（洗練された成功パターン） ===
        // 1. 現在の最強パターン（基準）
        vec![-3, 1, 1, -1, -5, -8, 8, 4, 2, 26],
        vec![-3, 1, 1, -1, -5, -8, 8, 4, 2, 26],
        vec![-3, 1, 1, -1, -5, -8, 8, 4, 2, 26],
        // 2. 現在の最強パターン微調整A（より安定化）
        vec![-3, 1, 1, -2, -5, -8, 7, 4, 2, 25],
        vec![-3, 1, 1, -2, -5, -8, 7, 4, 2, 25],
        vec![-3, 1, 1, -2, -5, -8, 7, 4, 2, 25],
        // 3. 現在の最強パターン微調整B（外周強化）
        vec![-3, 1, 1, -1, -5, -8, 8, 5, 3, 24],
        vec![-3, 1, 1, -1, -5, -8, 8, 5, 3, 24],
        vec![-3, 1, 1, -1, -5, -8, 8, 5, 3, 24],
        // === バランス重視グループ（安定性向上） ===
        // 4. 堅実バランス型A
        vec![-2, 1, 1, -2, -4, -7, 6, 3, 1, 22],
        vec![-2, 1, 1, -2, -4, -7, 6, 3, 1, 22],
        vec![-2, 1, 1, -2, -4, -7, 6, 3, 1, 22],
        // 5. 堅実バランス型B
        vec![-2, 0, 1, -1, -4, -7, 7, 4, 2, 23],
        vec![-2, 0, 1, -1, -4, -7, 7, 4, 2, 23],
        vec![-2, 0, 1, -1, -4, -7, 7, 4, 2, 23],
        // 6. 堅実バランス型C（55位パターン発展）
        vec![-1, 1, 0, -2, -4, -8, 5, 3, 1, 21],
        vec![-1, 1, 0, -2, -4, -8, 5, 3, 1, 21],
        // === 外周特化グループ（より洗練された外周戦略） ===
        // 7. 洗練外周型A
        vec![-3, 0, 0, -1, -5, -8, 8, 6, 4, 24],
        // 8. 洗練外周型B
        vec![-3, 0, 0, -2, -5, -8, 9, 6, 3, 25],
        // 9. 洗練外周型C
        vec![-2, 0, 1, -1, -4, -7, 7, 5, 3, 23],
        // === 中盤制御グループ（より精密な中盤戦略） ===
        // 10. 中盤抑制型A
        vec![-3, 1, 1, -2, -5, -9, 8, 4, 2, 25],
        // 11. 中盤抑制型B
        vec![-2, 1, 0, -2, -5, -8, 7, 4, 2, 24],
        // 12. 中盤抑制型C
        vec![-3, 0, 0, -1, -5, -9, 8, 5, 3, 26],
        // === ハイブリッドグループ（より効果的な組み合わせ） ===
        // 13. ハイブリッドA（現在最強×バランス）
        vec![-3, 1, 1, -1, -5, -8, 7, 4, 2, 24],
        // 14. ハイブリッドB（外周×中盤抑制）
        vec![-3, 0, 0, -2, -5, -8, 8, 5, 3, 25],
        // 15. ハイブリッドC（全要素ミックス）
        vec![-2, 1, 1, -2, -5, -8, 7, 4, 2, 23],
        // === 特殊戦略グループ（洗練された実験的アプローチ） ===
        // 16. 攻撃重視型
        vec![-3, 1, 1, -1, -5, -8, 9, 5, 3, 27],
        // 17. 防御重視型
        vec![-2, 1, 1, -2, -4, -7, 6, 3, 1, 22],
        // 18. エッジ支配型
        vec![-3, 0, 0, -1, -5, -8, 8, 6, 4, 25],
        // 19. 非対称型A（より洗練）
        vec![-3, 1, 1, -2, -5, -8, 8, 4, 2, 25],
        // 20. 非対称型B（より洗練）
        vec![-2, 0, 1, -1, -4, -8, 7, 5, 3, 24],
    ];

    let evaluator_vec: Vec<EvaluatorType<10>> = evaluators
        .into_iter()
        .map(|weights| {
            EvaluatorType::BitMatrix(Box::new(BitMatrixEvaluator::new(weights, masks.clone())))
        })
        .collect();
    let mut evaluator_pool = RankedEvaluatorPool::<10, 30>::new(evaluator_vec);

    let config = OptimizerConfig {
        population_size: 500,
        mutation_rate: 0.5,
        crossover_rate: 0.5,
        tournament_size: 25,
        max_generations: 100,
        early_stop_fitness: 0.80,
    };

    let max_loop = 100;
    for i in 0..max_loop {
        println!("loop: {}", i);
        let e_weights = evaluator_pool.get_weights();
        let fitness_calculator = MultiFitnessCalculator::<10>::new(
            evaluator_pool
                .iter()
                .zip(e_weights.iter())
                .map(|(eval, weight)| (eval.to_evaluator_type(), *weight))
                .collect(),
        );
        let mut optimizer = GeneticOptimizer::<10, GeneticMultiBitMatrixEvaluator<10>>::new(
            Box::new(fitness_calculator),
            config,
        );
        let best_evaluator = optimizer.optimize();
        evaluator_pool.push(best_evaluator.to_evaluator_type());
    }
}
