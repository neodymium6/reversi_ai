mod evaluator_evaluator;
mod evaluators;
mod fitness_calculator;
mod genetic_evaluators;
mod genetic_optimizer;
use fitness_calculator::bitmatrix::{EvaluatorType, MultiFitnessCalculator};
use genetic_evaluators::bitmatrix::GeneticBitMatrixEvaluator;
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
        // 2. 現在の最強パターン微調整A（より安定化）
        vec![-3, 1, 1, -2, -5, -8, 7, 4, 2, 25],
        // 3. 現在の最強パターン微調整B（外周強化）
        vec![-3, 1, 1, -1, -5, -8, 8, 5, 3, 24],
        // === バランス重視グループ（安定性向上） ===
        // 4. 堅実バランス型A
        vec![-2, 1, 1, -2, -4, -7, 6, 3, 1, 22],
        // 5. 堅実バランス型B
        vec![-2, 0, 1, -1, -4, -7, 7, 4, 2, 23],
        // 6. 堅実バランス型C（55位パターン発展）
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

    let weights = [
        0.10, // 1. 現在最強
        0.09, // 2. 最強微調整A
        0.09, // 3. 最強微調整B
        0.07, // 4. 堅実バランスA
        0.07, // 5. 堅実バランスB
        0.07, // 6. 堅実バランスC
        0.06, // 7. 外周型A
        0.06, // 8. 外周型B
        0.06, // 9. 外周型C
        0.05, // 10. 中盤抑制A
        0.05, // 11. 中盤抑制B
        0.05, // 12. 中盤抑制C
        0.04, // 13. ハイブリッドA
        0.04, // 14. ハイブリッドB
        0.04, // 15. ハイブリッドC
        0.02, // 16. 攻撃重視
        0.01, // 17. 防御重視
        0.01, // 18. エッジ支配
        0.01, // 19. 非対称A
        0.01, // 20. 非対称B
    ];
    let evaluator_vec: Vec<EvaluatorType<10>> = evaluators
        .into_iter()
        .map(|weights| {
            EvaluatorType::BitMatrix(Box::new(BitMatrixEvaluator::new(weights, masks.clone())))
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
        population_size: 300,
        mutation_rate: 0.2,
        crossover_rate: 0.5,
        tournament_size: 15,
        max_generations: 100,
    };
    let mut optimizer = GeneticOptimizer::<10, GeneticBitMatrixEvaluator<10>>::new(
        Box::new(fitness_calculator),
        config,
    );
    optimizer.optimize();
}
