use std::time::Duration;

use indicatif::ProgressBar;
use indicatif::ProgressState;
use indicatif::ProgressStyle;
use rand::Rng;
use rust_reversi_core::board::Board;
use rust_reversi_core::board::Turn;
use rust_reversi_core::search::AlphaBetaSearch;
use rust_reversi_core::search::Evaluator;

enum TurnOrder {
    Search1IsBlack,
    Search1IsWhite,
}

enum PlayResult {
    Search1Win,
    Search2Win,
    Draw,
}

pub struct EvaluatorEvaluator {
    search1: AlphaBetaSearch,
    search2: AlphaBetaSearch,
    timeout: Duration,
    epsilon: f64,
    show_progress: bool,
}

impl EvaluatorEvaluator {
    pub fn new(
        evaluator1: Box<dyn Evaluator>,
        evaluator2: Box<dyn Evaluator>,
        timeout: Duration,
        depth1: usize,
        depth2: usize,
        epsilon: f64,
        show_progress: bool,
    ) -> EvaluatorEvaluator {
        EvaluatorEvaluator {
            search1: AlphaBetaSearch::new(depth1, evaluator1),
            search2: AlphaBetaSearch::new(depth2, evaluator2),
            timeout,
            epsilon,
            show_progress,
        }
    }

    fn play_game_with_timeout(&self, timout: Duration, turn_order: TurnOrder) -> PlayResult {
        let search1_turn = match turn_order {
            TurnOrder::Search1IsBlack => Turn::Black,
            TurnOrder::Search1IsWhite => Turn::White,
        };
        let mut board = Board::new();
        let mut rng = rand::thread_rng();
        while !board.is_game_over() {
            if board.is_pass() {
                board.do_pass().unwrap();
                continue;
            }

            // for variety
            if rng.gen_bool(self.epsilon) {
                let action = board.get_random_move().unwrap();
                board.do_move(action).unwrap();
                continue;
            }

            if board.get_turn() == search1_turn {
                let action = self.search1.get_move_with_iter_deepening(&board, timout);
                let action = match action {
                    Some(action) => action,
                    None => {
                        eprintln!("Unknown error");
                        println!("{}", board);
                        panic!();
                    }
                };
                board.do_move(action).unwrap();
            } else {
                let action = self.search2.get_move_with_iter_deepening(&board, timout);
                let action = match action {
                    Some(action) => action,
                    None => {
                        eprintln!("Unknown error");
                        println!("{}", board);
                        panic!();
                    }
                };
                board.do_move(action).unwrap();
            }
        }
        let winner = board.get_winner().unwrap();
        match winner {
            Some(turn) => match turn == search1_turn {
                true => PlayResult::Search1Win,
                false => PlayResult::Search2Win,
            },
            None => PlayResult::Draw,
        }
    }

    fn play_game_with_depth(&self, turn_order: TurnOrder) -> PlayResult {
        let search1_turn = match turn_order {
            TurnOrder::Search1IsBlack => Turn::Black,
            TurnOrder::Search1IsWhite => Turn::White,
        };
        let mut board = Board::new();
        let mut rng = rand::thread_rng();
        while !board.is_game_over() {
            if board.is_pass() {
                board.do_pass().unwrap();
                continue;
            }

            // for variety
            if rng.gen_bool(self.epsilon) {
                let action = board.get_random_move().unwrap();
                board.do_move(action).unwrap();
                continue;
            }

            if board.get_turn() == search1_turn {
                let action = self.search1.get_move(&board);
                let action = match action {
                    Some(action) => action,
                    None => {
                        eprintln!("Unkown error");
                        println!("{}", board);
                        panic!();
                    }
                };
                board.do_move(action).unwrap();
            } else {
                let action = self.search2.get_move(&board);
                let action = match action {
                    Some(action) => action,
                    None => {
                        eprintln!("Unkown error");
                        println!("{}", board);
                        panic!();
                    }
                };
                board.do_move(action).unwrap();
            }
        }
        let winner = board.get_winner().unwrap();
        match winner {
            Some(turn) => match turn == search1_turn {
                true => PlayResult::Search1Win,
                false => PlayResult::Search2Win,
            },
            None => PlayResult::Draw,
        }
    }

    pub fn eval(&self, n: usize) -> (f64, f64) {
        let mut search1_win = 0;
        let mut search2_win = 0;
        let mut draw = 0;
        let pb = match self.show_progress {
            true => {
                let pb = ProgressBar::new(n as u64);
                pb.set_style(
                    ProgressStyle::with_template("[{wide_bar}] [{elapsed_precise}] ({eta})")
                        .unwrap()
                        .with_key(
                            "eta",
                            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                            },
                        )
                        .progress_chars("#>-"),
                );
                Some(pb)
            }
            false => None,
        };
        for _ in 0..n {
            match self.play_game_with_timeout(self.timeout, TurnOrder::Search1IsBlack) {
                PlayResult::Search1Win => search1_win += 1,
                PlayResult::Search2Win => search2_win += 1,
                PlayResult::Draw => draw += 1,
            }
            match self.play_game_with_timeout(self.timeout, TurnOrder::Search1IsWhite) {
                PlayResult::Search1Win => search1_win += 1,
                PlayResult::Search2Win => search2_win += 1,
                PlayResult::Draw => draw += 1,
            }
            match &pb {
                Some(pb) => pb.inc(1),
                None => {}
            }
        }
        let search1_win_rate = search1_win as f64 / (search1_win + search2_win + draw) as f64;
        let search2_win_rate = search2_win as f64 / (search1_win + search2_win + draw) as f64;
        (search1_win_rate, search2_win_rate)
    }

    pub fn eval_with_depth(&self, n: usize) -> (f64, f64) {
        let mut search1_win = 0;
        let mut search2_win = 0;
        let mut draw = 0;
        let pb = match self.show_progress {
            true => {
                let pb = ProgressBar::new(n as u64);
                pb.set_style(
                    ProgressStyle::with_template("[{wide_bar}] [{elapsed_precise}] ({eta})")
                        .unwrap()
                        .with_key(
                            "eta",
                            |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
                            },
                        )
                        .progress_chars("#>-"),
                );
                Some(pb)
            }
            false => None,
        };

        for _ in 0..n {
            match self.play_game_with_depth(TurnOrder::Search1IsBlack) {
                PlayResult::Search1Win => search1_win += 1,
                PlayResult::Search2Win => search2_win += 1,
                PlayResult::Draw => draw += 1,
            }
            match self.play_game_with_depth(TurnOrder::Search1IsWhite) {
                PlayResult::Search1Win => search1_win += 1,
                PlayResult::Search2Win => search2_win += 1,
                PlayResult::Draw => draw += 1,
            }
            match &pb {
                Some(pb) => pb.inc(1),
                None => {}
            }
        }
        let search1_win_rate = search1_win as f64 / (search1_win + search2_win + draw) as f64;
        let search2_win_rate = search2_win as f64 / (search1_win + search2_win + draw) as f64;
        (search1_win_rate, search2_win_rate)
    }
}
