use std::collections::HashMap;

use rust_reversi_core::search::Evaluator;

use crate::evaluator_evaluator::EvaluatorEvaluator;
use crate::fitness_calculator::bitmatrix::EvaluatorType;

#[derive(Clone)]
struct RankedEvaluator<const N: usize> {
    evaluator: EvaluatorType<N>,
    id: usize,
    rank_score: f64,
}

impl<const N: usize> RankedEvaluator<N> {
    fn new(evaluator: EvaluatorType<N>, id: usize, rank_score: f64) -> Self {
        RankedEvaluator {
            evaluator,
            id,
            rank_score,
        }
    }

    fn to_evaluator(&self) -> Box<dyn Evaluator> {
        self.evaluator.to_evaluator()
    }
}

pub struct RankedEvaluatorPool<const N: usize, const C: usize> {
    pool: [RankedEvaluator<N>; C],
    // cache for current items : (evaluator_id1, evaluator_id2) -> win_rate1
    results: HashMap<(usize, usize), f64>,
    // increment forever -> there is bug if it overflows
    // TODO: fix overflow bug. implement reset id (have to change cache key)
    next_id: usize,
    is_full: bool,
}

impl<const N: usize, const C: usize> RankedEvaluatorPool<N, C> {
    pub fn new(evals: Vec<EvaluatorType<N>>) -> Self {
        let mut instance = RankedEvaluatorPool {
            pool: std::array::from_fn(|i| {
                let rank_score = 0.0;
                RankedEvaluator::new(EvaluatorType::default(), i, rank_score) // place holder (overwritten by push)
            }),
            results: HashMap::new(),
            next_id: 0,
            is_full: false,
        };
        if evals.len() > C {
            eprintln!(
                "evaluator_pool: too many evaluators -> top {} will be used",
                C
            );
        }
        evals.iter().for_each(|eval| {
            instance.push(eval.clone());
        });
        instance
    }

    fn game_play_cached(
        &mut self,
        id1: usize,
        id2: usize,
        e1: Box<dyn Evaluator>,
        e2: Box<dyn Evaluator>,
    ) -> f64 {
        if let Some(result) = self.results.get(&(id1, id2)) {
            return *result;
        }
        let ee = EvaluatorEvaluator::new(
            e1,
            e2,
            std::time::Duration::from_millis(10),
            1,
            1,
            0.1,
            false,
        );
        let (score1, score2) = ee.eval_with_depth(100);
        self.results.insert((id1, id2), score1);
        self.results.insert((id2, id1), score2);
        score1
    }

    fn cache_clean_up(&mut self, id: usize) {
        self.results.retain(|(k1, k2), _| *k1 != id && *k2 != id);
    }

    pub fn push(&mut self, evaluator: EvaluatorType<N>) {
        let new_id = self.next_id;
        self.next_id += 1;
        let rank_score = 0.0;
        let mut new_evaluator = RankedEvaluator::new(evaluator, new_id, rank_score);
        if !self.is_full {
            self.pool[new_id] = new_evaluator;
            if self.next_id == C {
                self.is_full = true;
            }
            return;
        }
        let mut min_idx = 0;
        // rank score means the sum of win rate against others in the pool and new
        for i in 0..C {
            self.pool[i].rank_score = 0.0;
            for j in 0..C {
                if i == j {
                    continue;
                }
                let score = self.game_play_cached(
                    self.pool[i].id,
                    self.pool[j].id,
                    self.pool[i].to_evaluator(),
                    self.pool[j].to_evaluator(),
                );
                self.pool[i].rank_score += score;
            }

            // new vs old
            let score = self.game_play_cached(
                self.pool[i].id,
                new_id,
                self.pool[i].to_evaluator(),
                new_evaluator.to_evaluator(),
            );
            self.pool[i].rank_score += score;
            let score_new = self.game_play_cached(
                new_id,
                self.pool[i].id,
                new_evaluator.to_evaluator(),
                self.pool[i].to_evaluator(),
            );
            new_evaluator.rank_score += score_new;

            if self.pool[i].rank_score < self.pool[min_idx].rank_score {
                min_idx = i;
            }
        }
        if self.pool[min_idx].rank_score < new_evaluator.rank_score {
            // new is better than min -> replace
            self.cache_clean_up(self.pool[min_idx].id);
            self.pool[min_idx] = new_evaluator;
        } else {
            self.cache_clean_up(new_id);
        }
    }
}
