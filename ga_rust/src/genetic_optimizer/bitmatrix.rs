use std::sync::Arc;
use std::sync::Mutex;

use rand::Rng;

use crate::evaluators::bitmatrix::GeneticBitMatrixEvaluator;
use crate::fitness_calculator::bitmatrix::SimpleFitnessCalculator;
use crate::genetic_optimizer::OptimizerConfig;
use indicatif::ProgressBar;
use indicatif::ProgressState;
use indicatif::ProgressStyle;
use rayon::prelude::*;

pub struct BitMatrixOptimizer<const N: usize> {
    population: Vec<GeneticBitMatrixEvaluator<N>>,
    fitness_calculator: SimpleFitnessCalculator<N>,
    generation: usize,
    config: OptimizerConfig,
}

impl<const N: usize> BitMatrixOptimizer<N> {
    pub fn new(
        fitness_calculator: SimpleFitnessCalculator<N>,
        config: OptimizerConfig,
    ) -> BitMatrixOptimizer<N> {
        let mut population = Vec::new();
        for _ in 0..config.population_size {
            population.push(GeneticBitMatrixEvaluator::<N>::new_from_random());
        }
        BitMatrixOptimizer {
            population,
            fitness_calculator,
            generation: 0,
            config,
        }
    }

    fn evaluate_fitness(&self) -> Vec<f64> {
        let pb = ProgressBar::new(self.config.population_size as u64);
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
        let pb = Arc::new(Mutex::new(pb));
        self.population
            .par_iter()
            // .iter()
            .map(|evaluator| {
                self.fitness_calculator
                    .calculate_fitness(evaluator.to_evaluator())
            })
            .inspect(|_| {
                let pb = pb.lock().unwrap();
                pb.inc(1);
            })
            .collect()
    }

    fn select(&self, fitnesses: Vec<f64>) -> Vec<GeneticBitMatrixEvaluator<N>> {
        let mut rng = rand::thread_rng();
        let mut selected = Vec::new();
        for _ in 0..self.config.population_size {
            let mut best_index = 0;
            let mut best_fitness = 0.0;
            for _i in 0..self.config.tournament_size {
                let index = rng.gen_range(0..self.config.population_size);
                if fitnesses[index] > best_fitness {
                    best_index = index;
                    best_fitness = fitnesses[index];
                }
            }
            selected.push(self.population[best_index].clone());
        }
        selected
    }

    fn evolve(&mut self, fitnesses: Vec<f64>) {
        let selected = self.select(fitnesses);
        let mut new_population = Vec::new();
        for i in 0..self.config.population_size {
            let mut rng = rand::thread_rng();
            let parent1 = match rng.gen_bool(self.config.mutation_rate) {
                true => &selected[i].mutate(),
                false => &selected[i],
            };
            match rng.gen_bool(self.config.crossover_rate) {
                true => {
                    let parent2 = selected[rng.gen_range(0..self.config.population_size)].clone();
                    new_population.push(parent1.crossover(&parent2));
                }
                false => new_population.push(parent1.clone()),
            }
        }
        self.population = new_population;
    }

    pub fn optimize(&mut self) -> GeneticBitMatrixEvaluator<N> {
        let mut global_best_indivisual = GeneticBitMatrixEvaluator::<N>::new_from_random();
        let mut global_best_fitness = 0.0;
        for _ in 0..self.config.max_generations {
            let mut best_index = 0;
            let mut best_fitness = 0.0;
            let fitnesses = self.evaluate_fitness();
            for (i, &fitness) in fitnesses.iter().enumerate() {
                if fitness > best_fitness {
                    best_index = i;
                    best_fitness = fitness;
                }
            }
            if best_fitness > global_best_fitness {
                global_best_indivisual = self.population[best_index].clone();
                global_best_fitness = best_fitness;
            }
            println!(
                "Generation: {}, Best Fitness: {}",
                self.generation, best_fitness
            );
            println!("Best Individual: {:?}", self.population[best_index]);
            self.evolve(fitnesses);
            self.generation += 1;
        }
        global_best_indivisual
    }
}
