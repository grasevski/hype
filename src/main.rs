//! Command line hyperparameter tuner.
use chrono::offset::Utc;
use clap::Clap;
use csv::{Reader, Writer};
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    convert::{TryFrom, TryInto},
    error::Error,
    io,
    io::Write,
    process,
};
use tinyvec::ArrayVec;
use tpe::{
    categorical_range, density_estimation::DefaultEstimatorBuilder, histogram_estimator,
    parzen_estimator, range, range::RangeError, TpeOptimizer,
};

/// A hyperparameter to be optimized.
#[derive(Deserialize)]
enum Param {
    /// Discrete variable.
    Categorical(usize),

    /// Continuous variable.
    Numeric(f64, f64),
}

/// Hyperparameter data.
struct ParamState {
    /// Current value for this parameter.
    value: f64,

    /// Whether the parameter is categorical.
    categorical: bool,

    /// Optimizer for this parameter.
    optim: TpeOptimizer<DefaultEstimatorBuilder>,
}

impl TryFrom<Param> for ParamState {
    type Error = RangeError;

    fn try_from(param: Param) -> Result<Self, RangeError> {
        let (categorical, optim) = match param {
            Param::Categorical(param) => (
                true,
                TpeOptimizer::new(histogram_estimator(), categorical_range(param)?),
            ),
            Param::Numeric(lower, upper) => (
                false,
                TpeOptimizer::new(parzen_estimator(), range(lower, upper)?),
            ),
        };
        Ok(Self {
            value: Default::default(),
            categorical,
            optim,
        })
    }
}

impl ParamState {
    /// Get next hyperparam to try.
    fn ask(&mut self, rng: &mut impl Rng) {
        self.value = self.optim.ask(rng).expect("ask failed");
    }

    /// Update optimizer with result.
    fn tell(&mut self, score: f64) -> Result<(), impl Error> {
        self.optim.tell(self.value, score)
    }

    /// Get the current hyperparam.
    fn get_value(&self) -> String {
        if self.categorical {
            (self.value as usize).to_string()
        } else {
            self.value.to_string()
        }
    }
}

/// Run hyperparameter optimization on the given command.
#[derive(Clap)]
struct Opts {
    /// Json dict of hyperparameters to be optimized.
    params: String,

    /// Command to run one iteration of hyperparam tuning.
    cmd: String,

    /// Fixed arguments to command.
    args: Vec<String>,

    /// Number of tuning iterations.
    #[clap(short, long, default_value = "100")]
    iter: u32,

    /// Random number generator seed.
    #[clap(short, long, default_value = "0")]
    seed: u64,

    /// Whether to maximize objective function.
    #[clap(short, long)]
    maximize: bool,
}

/// Runs the hyperparameter optimization.
fn main() -> Result<(), Box<dyn Error>> {
    let opts = Opts::parse();
    let params: Result<BTreeMap<String, Param>, _> = serde_json::from_str(&opts.params);
    let params: Result<BTreeMap<String, ParamState>, RangeError> = params?
        .into_iter()
        .map(|(k, v)| Ok((k, v.try_into()?)))
        .collect();
    let mut params = params?;
    let mut rng = StdRng::seed_from_u64(opts.seed);
    let mut wtr = Writer::from_writer(io::stdout());
    let mut has_header = false;
    for i in 0..opts.iter {
        for v in params.values_mut() {
            v.ask(&mut rng);
        }
        let output = process::Command::new(opts.cmd.clone())
            .args(
                opts.args
                    .clone()
                    .into_iter()
                    .chain(params.iter().flat_map(|(k, v)| {
                        ArrayVec::from([
                            format!("{}{}", if k.len() == 1 { "-" } else { "--" }, k),
                            v.get_value(),
                        ])
                        .into_iter()
                    })),
            )
            .output()?;
        io::stderr().write_all(&output.stderr[..])?;
        io::stderr().flush()?;
        let ts = Utc::now();
        if !output.status.success() {
            let params: BTreeMap<String, String> = params
                .into_iter()
                .map(|(k, v)| (k, v.get_value()))
                .collect();
            panic!(
                "Subprocess exited with status {} on iteration {} at {}. Variables: {:?}",
                output.status, i, ts, params
            );
        }
        let mut best = f64::INFINITY;
        let mut rdr = Reader::from_reader(&output.stdout[..]);
        let headers = rdr.headers()?;
        let score = headers
            .iter()
            .position(|x| x == "score")
            .expect("score missing");
        if !has_header {
            wtr.write_record(
                ArrayVec::from(["ts".to_string(), "iter".to_string()])
                    .into_iter()
                    .chain(params.keys().cloned())
                    .chain(headers.iter().map(|x| x.to_string())),
            )?;
            wtr.flush()?;
            has_header = true;
        }
        for row in Reader::from_reader(&output.stdout[..]).into_records() {
            let row = row?;
            let score: f64 = row[score].parse()?;
            best = best.min(score * if opts.maximize { -1.0 } else { 1.0 });
            wtr.write_record(
                ArrayVec::from([ts.to_string(), i.to_string()])
                    .into_iter()
                    .chain(params.values().map(ParamState::get_value))
                    .chain(row.iter().map(|x| x.to_string())),
            )?;
            wtr.flush()?;
        }
        for v in params.values_mut() {
            v.tell(best)?;
        }
    }
    Ok(())
}
