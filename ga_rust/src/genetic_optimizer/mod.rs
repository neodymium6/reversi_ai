pub mod bitmatrix;

pub struct OptimizerConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub tournament_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
}

// impl OptimizerConfig {
//     pub fn new(
//         population_size: usize,
//         max_generations: usize,
//         tournament_size: usize,
//         mutation_rate: f64,
//         crossover_rate: f64,
//         selection_rate: f64,
//     ) -> OptimizerConfig {
//         OptimizerConfig {
//             population_size,
//             max_generations,
//             tournament_size,
//             mutation_rate,
//             crossover_rate,
//             selection_rate,
//         }
//     }
// }
