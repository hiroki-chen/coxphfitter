//! This crate implements a simple Cox proportional hazards model for survival analysis.

#![allow(non_snake_case)]

use anyhow::{anyhow, Result};
use argmin::core::{CostFunction, Executor, Gradient, Hessian, Jacobian, Operator as ArgminOp};
use argmin::solver::linesearch::condition::ArmijoCondition;
use argmin::solver::linesearch::BacktrackingLineSearch;
use argmin::solver::quasinewton::BFGS;
use ndarray::prelude::*;
use polars::frame::DataFrame;
use polars::prelude::{DataType, Float64Type, IndexOrder};

/// The input arguments for the Cox proportional hazards model.
#[derive(Debug, Clone)]
pub struct CoxPHFitterArgs<'a> {
    /// Attach a penalty to the size of the coefficients during regression.
    /// This improves stability of the estimates and controls for high correlation between covariates.
    ///
    /// ```tex
    ///     $penalizer(\dfrac{1 - l1\_ratio}{2} || \beta ||^{2}_{2} + l1\_ratio ||\beta||_1 )$
    /// ```
    pub penalizer: f64,
    /// Specify how the fitter should estimate the baseline.
    pub baseline_estimation_method: &'a str,
    /// Specify what raio to assign to a L1 vs L2 penalty. Will default to 0.0 if not specified.
    pub l1_ratio: f64,
    /// The name of the column in DataFrame that contains the subjects’ lifetimes.
    pub duration_col: &'a str,
    /// The name of the column in DataFrame that contains the subjects’ death observation.
    pub event_col: &'a str,
    /// The columns to use as covariates in the model.
    pub robust: bool,
}

impl<'a> CoxPHFitterArgs<'a> {
    /// Creates a new instance of `CoxPHFitterArgs` with the provided parameters.
    pub fn new(
        penalizer: f64,
        baseline_estimation_method: &'a str,
        l1_ratio: f64,
        duration_col: &'a str,
        event_col: &'a str,
        robust: bool,
    ) -> Self {
        Self {
            penalizer,
            baseline_estimation_method,
            l1_ratio,
            duration_col,
            event_col,
            robust,
        }
    }

    pub fn set_penalizer(mut self, penalizer: f64) -> Self {
        self.penalizer = penalizer;
        self
    }

    pub fn set_baseline_estimation_method(mut self, method: &'a str) -> Self {
        self.baseline_estimation_method = method;
        self
    }

    pub fn set_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    pub fn set_duration_col(mut self, duration_col: &'a str) -> Self {
        self.duration_col = duration_col;
        self
    }

    pub fn set_event_col(mut self, event_col: &'a str) -> Self {
        self.event_col = event_col;
        self
    }

    pub fn set_robust(mut self, robust: bool) -> Self {
        self.robust = robust;
        self
    }
}

impl<'a> Default for CoxPHFitterArgs<'a> {
    fn default() -> Self {
        Self::new(
            // Default values for the CoxPHFitterArgs
            0.0,        // penalizer
            "breslow",  // baseline_estimation_method
            0.0,        // l1_ratio
            "duration", // duration_col
            "event",    // event_col
            false,      // robust
        )
    }
}

/// This class implements fitting Cox’s proportional hazard model.
///
/// ```tex
///     $h(t \mid x) = h_0(t) \exp((x \overline{x})'\beta)$
/// ```
///
/// The baseline hazard can be modeled in two ways:
///
/// 1. (default) non-parametrically, using Breslow’s method. In this case, the entire model is the
///     traditional semi-parametric Cox model. Ties are handled using Efron’s method.
/// 2. parametrically, using a pre-specified number of cubic splines, or piecewise values. (on-going work)
#[derive(Debug, Clone)]
pub struct CoxPHFitter<'a> {
    args: CoxPHFitterArgs<'a>,
    time_to_event: Option<Array1<f64>>,
    event_indicator: Option<Array1<f64>>,
    covariates_matrix: Option<Array2<f64>>,
}

impl<'a> CoxPHFitter<'a> {
    /// Creates a new instance of `CoxPHFitter` with the provided arguments.
    #[inline]
    pub fn new(args: CoxPHFitterArgs<'a>) -> Self {
        Self {
            args,
            time_to_event: None,
            event_indicator: None,
            covariates_matrix: None,
        }
    }

    /// Fit the Cox proportional hazard model to a dataset.
    pub fn fit(&mut self, df: &DataFrame) -> Result<CoxPHResults> {
        let time_array = df
            .column(self.args.duration_col)?
            .cast(&DataType::Float64)?
            .f64()?
            .into_no_null_iter()
            .collect::<Array1<f64>>();
        let event_array = df
            .column(self.args.event_col)?
            .cast(&DataType::Float64)?
            .f64()?
            .into_no_null_iter()
            .collect::<Array1<f64>>();
        let covariates_df = df.drop_many(&[self.args.duration_col, self.args.event_col]);
        let num_covariates = covariates_df.width();
        let covariates_array = covariates_df
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)?
            // In general, using the Fortran-like index order is faster.
            .into_shape((covariates_df.height(), num_covariates))
            .map_err(|_| anyhow!("Failed to convert DataFrame to 2D ndarray"))?;

        // Store data in the fitter for argmin
        self.time_to_event = Some(time_array);
        self.event_indicator = Some(event_array);
        self.covariates_matrix = Some(covariates_array);

        let initial_beta = Array1::<f64>::zeros(num_covariates);
        let initial_inv_hessian = Array2::<f64>::eye(num_covariates);
        let solver = BFGS::new(BacktrackingLineSearch::new(
            ArmijoCondition::new(1e-4).unwrap(),
        ));

        let res = Executor::new(self.clone(), solver) // Clone self because Executor takes ownership of Op
            .configure(|state| {
                state
                    .param(initial_beta)
                    .inv_hessian(initial_inv_hessian)
                    .max_iters(1000)
                    .target_cost(1e-6)
            }) // Increased max_iters, added target_cost
            .run()?;

        // --- Extract and Process Results ---
        let final_beta = res.state.param.unwrap(); // Optimized coefficients
        let final_log_likelihood = -res.state.cost; // `argmin` minimizes cost, so take negative for log-likelihood
        let num_iterations = res.state.iter;

        Ok(CoxPHResults {
            coefficients: final_beta,
            num_iterations,
            final_log_likelihood,
            ..Default::default()
        })
    }
}

impl<'a> ArgminOp for CoxPHFitter<'a> {
    type Param = Array1<f64>;

    // The negative penalized partial log-likelihood
    type Output = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output> {
        // Retrieve data from self.
        // Unwrap is fine here because `fit` method will ensure these are `Some`.
        let time_to_event = self.time_to_event.as_ref().unwrap();
        let event_indicator = self.event_indicator.as_ref().unwrap();
        let covariates_matrix = self.covariates_matrix.as_ref().unwrap();

        let num_observations = time_to_event.len();
        let mut neg_log_likelihood = 0.0;

        // Iterate through unique event times to build risk sets
        // This is a simplified loop, proper handling of tied events (Breslow/Efron)
        // is crucial here and impacts gradient/hessian too.
        let mut sorted_indices: Vec<usize> = (0..num_observations).collect();
        sorted_indices.sort_by(|&a, &b| time_to_event[a].partial_cmp(&time_to_event[b]).unwrap());

        for &i in sorted_indices.iter() {
            if event_indicator[i] == 1.0 {
                // Only consider observed events
                let current_time = time_to_event[i];

                // Calculate risk set for the current event time
                // Risk set includes all individuals whose time_to_event >= current_time
                let risk_set_indices: Vec<usize> = (0..num_observations)
                    .filter(|&j| time_to_event[j] >= current_time)
                    .collect();

                let numerator_sum = (covariates_matrix.row(i).to_owned().dot(param)).exp();

                let mut denominator_sum = 0.0;
                for &k_idx in risk_set_indices.iter() {
                    denominator_sum += (covariates_matrix.row(k_idx).to_owned().dot(param)).exp();
                }

                // Add to negative log-likelihood (for maximization, we minimize negative)
                if denominator_sum > 0.0 {
                    neg_log_likelihood -= (numerator_sum / denominator_sum).ln();
                }
            }
        }

        // Add penalization (L1 and L2 terms)
        let l1_penalty = param.mapv(f64::abs).sum();
        let l2_penalty = param.dot(param); // ||beta||^2_2

        let penalty_term = self.args.penalizer
            * ((1.0 - self.args.l1_ratio) / 2.0 * l2_penalty + self.args.l1_ratio * l1_penalty);

        Ok(neg_log_likelihood + penalty_term)
    }
}
impl<'a> CostFunction for CoxPHFitter<'a> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output> {
        let time_to_event = self.time_to_event.as_ref().unwrap();
        let event_indicator = self.event_indicator.as_ref().unwrap();
        let covariates_matrix = self.covariates_matrix.as_ref().unwrap();

        let num_observations = time_to_event.len();
        let mut neg_log_likelihood = 0.0;

        // Create a sorted list of indices based on time_to_event
        let mut sorted_indices: Vec<usize> = (0..num_observations).collect();
        sorted_indices.sort_by(|&a, &b| time_to_event[a].partial_cmp(&time_to_event[b]).unwrap());

        for &i in sorted_indices.iter() {
            if event_indicator[i] == 1.0 {
                // Only consider events
                let current_time = time_to_event[i];

                // Determine the risk set: all individuals alive at or after current_time
                let risk_set_indices: Vec<usize> = (0..num_observations)
                    .filter(|&j| time_to_event[j] >= current_time)
                    .collect();

                let numerator_exp_beta_x = (covariates_matrix.row(i).dot(param)).exp();

                let mut denominator_sum_exp_beta_x = 0.0;
                for &k_idx in risk_set_indices.iter() {
                    denominator_sum_exp_beta_x += (covariates_matrix.row(k_idx).dot(param)).exp();
                }

                if denominator_sum_exp_beta_x <= 0.0 {
                    // Return a very large value to push optimizer away from this degenerate region
                    return Ok(f64::INFINITY);
                }
                neg_log_likelihood -= (numerator_exp_beta_x / denominator_sum_exp_beta_x).ln();
            }
        }

        // Add L1 and L2 penalties
        let l1_penalty = param.mapv(f64::abs).sum();
        let l2_penalty = param.dot(param);

        let penalty_term = self.args.penalizer
            * ((1.0 - self.args.l1_ratio) / 2.0 * l2_penalty + self.args.l1_ratio * l1_penalty);

        Ok(neg_log_likelihood + penalty_term)
    }
}

// 2. Implement the Gradient trait
impl<'a> Gradient for CoxPHFitter<'a> {
    type Param = Array1<f64>;
    type Gradient = Array1<f64>; // Output type for gradient is usually the same as Param

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient> {
        let time_to_event = self.time_to_event.as_ref().unwrap();
        let event_indicator = self.event_indicator.as_ref().unwrap();
        let covariates_matrix = self.covariates_matrix.as_ref().unwrap();

        let num_observations = time_to_event.len();
        let num_covariates = param.len();
        let mut grad = Array1::zeros(num_covariates);

        let mut sorted_indices: Vec<usize> = (0..num_observations).collect();
        sorted_indices.sort_by(|&a, &b| time_to_event[a].partial_cmp(&time_to_event[b]).unwrap());

        for &i in sorted_indices.iter() {
            if event_indicator[i] == 1.0 {
                let current_time = time_to_event[i];

                let risk_set_indices: Vec<usize> = (0..num_observations)
                    .filter(|&j| time_to_event[j] >= current_time)
                    .collect();

                let mut sum_exp_beta_x = 0.0;
                let mut sum_x_exp_beta_x = Array1::zeros(num_covariates);

                for &k_idx in risk_set_indices.iter() {
                    let exp_beta_x = (covariates_matrix.row(k_idx).dot(param)).exp();
                    sum_exp_beta_x += exp_beta_x;
                    sum_x_exp_beta_x += &(covariates_matrix.row(k_idx).to_owned() * exp_beta_x);
                }

                if sum_exp_beta_x > 0.0 {
                    let expected_x_in_risk_set = sum_x_exp_beta_x / sum_exp_beta_x;
                    grad -= &(covariates_matrix.row(i).to_owned() - expected_x_in_risk_set);
                }
            }
        }

        // L1 regularization adds signum of parameter
        let l1_grad = param.mapv(f64::signum);
        let l2_grad = 2.0 * param;

        let penalty_grad = self.args.penalizer
            * ((1.0 - self.args.l1_ratio) / 2.0 * l2_grad + self.args.l1_ratio * l1_grad);
        grad += &penalty_grad;

        Ok(grad)
    }
}

// 3. Implement the Hessian trait
impl<'a> Hessian for CoxPHFitter<'a> {
    type Param = Array1<f64>;
    type Hessian = Array2<f64>; // Output type for Hessian is Array2

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian> {
        let time_to_event = self.time_to_event.as_ref().unwrap();
        let event_indicator = self.event_indicator.as_ref().unwrap();
        let covariates_matrix = self.covariates_matrix.as_ref().unwrap();

        let num_observations = time_to_event.len();
        let num_covariates = param.len();
        let mut hess = Array2::zeros((num_covariates, num_covariates));

        let mut sorted_indices: Vec<usize> = (0..num_observations).collect();
        sorted_indices.sort_by(|&a, &b| time_to_event[a].partial_cmp(&time_to_event[b]).unwrap());

        for &i in sorted_indices.iter() {
            if event_indicator[i] == 1.0 {
                let current_time = time_to_event[i];
                let risk_set_indices: Vec<usize> = (0..num_observations)
                    .filter(|&j| time_to_event[j] >= current_time)
                    .collect();

                let mut sum_exp_beta_x = 0.0;
                let mut sum_x_exp_beta_x = Array1::zeros(num_covariates);
                let mut sum_xxT_exp_beta_x = Array2::zeros((num_covariates, num_covariates));

                for &k_idx in risk_set_indices.iter() {
                    let x_k: ArrayView1<f64> = covariates_matrix.row(k_idx); // Use ArrayView1 for efficiency
                    let exp_beta_x = (x_k.dot(param)).exp();
                    sum_exp_beta_x += exp_beta_x;
                    sum_x_exp_beta_x += &(x_k.to_owned() * exp_beta_x);
                    sum_xxT_exp_beta_x +=
                        &(x_k.insert_axis(Axis(1)).dot(&x_k.insert_axis(Axis(0))) * exp_beta_x);
                }

                if sum_exp_beta_x > 0.0 {
                    let expected_x = sum_x_exp_beta_x / sum_exp_beta_x;
                    let expected_xxT = sum_xxT_exp_beta_x / sum_exp_beta_x;

                    let outer_product_of_expected_x = expected_x
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&expected_x.insert_axis(Axis(0)));
                    hess += &(expected_xxT - outer_product_of_expected_x);
                }
            }
        }
        // Add L2 penalty part to Hessian (second derivative of 0.5 * penalizer * l2_penalty * beta^2)
        let l2_hess =
            Array2::eye(num_covariates) * (self.args.penalizer * (1.0 - self.args.l1_ratio));
        hess += &l2_hess;

        Ok(hess)
    }
}

impl<'a> Jacobian for CoxPHFitter<'a> {
    // Not used for this type of problem
    type Jacobian = ();
    type Param = Array1<f64>;

    fn jacobian(&self, _param: &Self::Param) -> Result<Self::Jacobian> {
        Ok(())
    }
}

/// Represents the fitted Cox Proportional Hazards Model results.
#[derive(Default, Debug)]
pub struct CoxPHResults {
    pub coefficients: Array1<f64>,
    pub hazard_ratios: Array1<f64>,
    pub standard_errors: Array1<f64>,
    pub p_values: Array1<f64>,
    pub log_likelihood: f64,
    pub num_iterations: u64,
    pub convergence_succeeded: bool,
    pub covariate_names: Vec<String>,
    pub final_log_likelihood: f64,
    // You might also store the baseline hazard/survival function here
    // pub baseline_hazard: Vec<(f64, f64)>, // (time, hazard)
    // pub baseline_survival: Vec<(f64, f64)>, // (time, survival_probability)
}

#[cfg(test)]
mod test {
    use polars::prelude::LazyFrame;

    use super::*;

    const TEST_DF_PATH: &str = "./data/cox_data.parquet";

    #[test]
    fn test_fit() {
        let args = CoxPHFitterArgs::default()
            .set_event_col("event")
            .set_duration_col("T")
            .set_penalizer(0.1)
            .set_robust(true);
        let mut fitter = CoxPHFitter::new(args);

        // Load a test DataFrame (this is a placeholder, replace with actual DataFrame loading logic)
        let df = LazyFrame::scan_parquet(TEST_DF_PATH, Default::default())
            .unwrap()
            .collect()
            .unwrap();

        // Fit the Cox proportional hazards model
        let result = fitter.fit(&df);

        assert!(
            result.is_ok(),
            "Failed to fit CoxPHFitter: {:?}",
            result.err()
        );

        println!("Fitted CoxPHFitter successfully: {:#?}", result.unwrap());
    }
}
