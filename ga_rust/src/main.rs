mod evaluator_evaluator;
use evaluator_evaluator::EvaluatorEvaluator;
use rust_reversi_core::search::LegalNumEvaluator;
use rust_reversi_core::search::PieceEvaluator;

const TIMEOUT: std::time::Duration = std::time::Duration::from_millis(5);
const EPSILON: f64 = 5e-2;
fn main() {
    let evaluator1 = Box::new(LegalNumEvaluator::new());
    let evaluator2 = Box::new(PieceEvaluator::new());
    let evaluator_evaluator = EvaluatorEvaluator::new(evaluator1, evaluator2, TIMEOUT, EPSILON);
    let result = evaluator_evaluator.eval(100);
    println!("{:?}", result);
}
