use crate::evaluators::{GeneticEvaluator, GeneticEvaluatorFactory};
use crate::fitness_calculator::FitnessCalculator;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use rand::Rng;
use rust_reversi_core::search::Evaluator;

pub struct GeneticRateConfig {
    mutation_rate: f64,
    crossover_rate: f64,
    selection_rate: f64,
}

impl GeneticRateConfig {
    pub fn new(mutation_rate: f64, crossover_rate: f64, selection_rate: f64) -> GeneticRateConfig {
        GeneticRateConfig {
            mutation_rate,
            crossover_rate,
            selection_rate,
        }
    }
}

pub struct GeneticOptimizer<Y: GeneticEvaluatorFactory, Z: FitnessCalculator> {
    population: Vec<Box<dyn GeneticEvaluator>>,
    population_size: usize,
    generation: usize,
    rate_config: GeneticRateConfig,
    tournament_size: usize,
    max_generations: usize,
    factory: Y,
    fitness_calculator: Z,
}

impl<Y: GeneticEvaluatorFactory, Z: FitnessCalculator> GeneticOptimizer<Y, Z> {
    fn initialize_population(size: usize, factory: &Y) -> Vec<Box<dyn GeneticEvaluator>> {
        let mut population = Vec::new();
        for _ in 0..size {
            population.push(factory.generate());
        }
        population
    }

    pub fn new(
        population_size: usize,
        rate_config: GeneticRateConfig,
        tournament_size: usize,
        max_generations: usize,
        factory: Y,
        fitness_calculator: Z,
    ) -> GeneticOptimizer<Y, Z> {
        GeneticOptimizer {
            population: GeneticOptimizer::<Y, Z>::initialize_population(population_size, &factory),
            population_size,
            generation: 0,
            rate_config,
            tournament_size,
            max_generations,
            factory,
            fitness_calculator,
        }
    }

    fn evaluate_fitness(&self) -> Vec<f64> {
        let pb = ProgressBar::new(self.population_size as u64);
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
        let mut fitnesses = Vec::new();
        for individual in &self.population {
            fitnesses.push(
                self.fitness_calculator
                    .calculate_fitness(individual.to_evaluator()),
            );
            pb.inc(1);
        }
        fitnesses
    }

    fn select_parent(&self, fitnesses: &[f64]) -> &dyn GeneticEvaluator {
        let mut rng = rand::thread_rng();
        let mut best = rng.gen_range(0..self.population_size);
        for _ in 0..self.tournament_size {
            let candidate = rng.gen_range(0..self.population_size);
            if fitnesses[candidate] > fitnesses[best] {
                best = candidate;
            }
        }
        self.population[best].as_ref()
    }

    fn evolve(&mut self, fitnesses: &[f64]) {
        let mut new_population = Vec::new();
        for _ in 0..self.population_size {
            let parent1 = self.select_parent(fitnesses);
            let parent2 = self.select_parent(fitnesses);
            let mut rng = rand::thread_rng();
            let child = if rng.gen_bool(self.rate_config.crossover_rate) {
                parent1.crossover(parent2)
            } else {
                parent1.mutate()
            };
            new_population.push(child);
        }
        self.population = new_population;
        self.generation += 1;
    }

    pub fn optimize(&mut self) -> Box<dyn Evaluator> {
        let mut best = 0;
        let mut best_fitness = 0.0;
        for _ in 0..self.max_generations {
            let fitnesses = self.evaluate_fitness();
            for (i, &fitness) in fitnesses.iter().enumerate() {
                if fitness > best_fitness {
                    best = i;
                    best_fitness = fitness;
                }
            }
            println!(
                "Generation: {}, Best Fitness: {}",
                self.generation, best_fitness
            );
            println!("Best Individual: {:?}", self.population[best]);
            self.evolve(&fitnesses);
        }
        self.population[best].to_evaluator()
    }
}
