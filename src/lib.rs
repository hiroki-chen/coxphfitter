//! This crate implements a simple Cox proportional hazards model for survival analysis.
//! does not work.
//!
//! Try survival-rust first.

#![allow(non_snake_case)]

pub(crate) mod math;

use anyhow::{anyhow, Result};
use math::{elastic_net_penalty, StepSizer};
use ndarray::prelude::*;
use ndarray_einsum_beta::einsum;
use ndarray_linalg::*;
use polars::frame::DataFrame;
use polars::prelude::*;

#[derive(Debug, Clone)]
pub struct Coefficient {
    /// The coefficient
    pub coef: f64,
    /// Thwe exponentiated coefficient value estimate
    pub exp_coef: f64,
    /// The standard error of the coefficient estimate
    pub se_coef: f64,
    /// The Z statistic, which is the ratio of the coefficient estimate to its standard error
    pub z: f64,
}

/// The input arguments for the Cox proportional hazards model.
#[derive(Debug, Clone)]
pub struct CoxPHFitterArgs<'a> {
    /// Attach a penalty to the size of the coefficients during regression.
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
    /// Smoothing parameter for the soft_abs function in L1 penalty.
    pub a_param_soft_abs: f64,
    /// Max iteration
    pub max_iter: usize,
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
        max_iter: usize,
    ) -> Self {
        Self {
            penalizer,
            baseline_estimation_method,
            l1_ratio,
            duration_col,
            event_col,
            robust,
            a_param_soft_abs: 100.0, // Default smoothing parameter for L1
            max_iter,
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

    pub fn set_a_param_soft_abs(mut self, a_param: f64) -> Self {
        self.a_param_soft_abs = a_param;
        self
    }

    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }
}

impl<'a> Default for CoxPHFitterArgs<'a> {
    fn default() -> Self {
        Self::new(0.0, "breslow", 0.0, "duration", "event", false, 500)
    }
}

/// This class implements fitting Cox’s proportional hazard model.
///
/// Cox proportional hazards models are the most widely used approach to modeling time to even data.
/// The hazard function computes the instantaneous rate of an event occurrence, written $h(t)$.
///
/// The hazard function for aobservation $i$ in a Cox PH model is defined as ℎ𝑖(𝑡)=𝜆(𝑡)exp(𝐱𝑇𝑖𝛽)
#[derive(Debug, Clone)]
pub struct CoxPHFitter<'a> {
    args: CoxPHFitterArgs<'a>,
    time_to_event: Option<Array1<f64>>,
    event_indicator: Option<Array1<f64>>, // Should store bool or 0/1
    covariates_matrix: Option<Array2<f64>>,
    // Store sorted indices and original data to avoid repeated sorting if data doesn't change
    // These would be populated during a `preprocess` step or at the start of `fit`
    sorted_original_indices: Option<Vec<usize>>,
}

/// Represents the fitted Cox Proportional Hazards Model results.
#[derive(Default, Debug, Clone)]
pub struct CoxPHResults {
    /// The coefficients are a list of the estimated coefficients for each covariate in the model.
    /// We also store the standard eror of the coefficient estimate.
    pub coefficients: Array1<f64>,
    pub hazard_ratios: Array1<f64>,
    pub standard_errors: Array1<f64>,
    pub p_values: Array1<f64>,
    pub log_likelihood: f64, // Actual log-likelihood (not negative)
    pub num_iterations: u64,
    pub convergence_succeeded: bool, // Placeholder, argmin result doesn't directly give this boolean
    pub covariate_names: Vec<String>, // Placeholder
    pub final_log_likelihood: f64,   // Kept for compatibility, same as log_likelihood
}

pub struct CoxProcessedDf {
    /// The time to event column, converted to an Array1<f64>
    time_array: Array1<f64>,
    /// The event indicator column, converted to an Array1<i32>
    event_array: Array1<i32>,
    /// The covariates matrix, converted to an Array2<f64>
    covariates: Array2<f64>,
    /// Weights.
    weights: Array1<f64>,
}

impl<'a> CoxPHFitter<'a> {
    fn preprocess(&self, df: &DataFrame) -> Result<CoxProcessedDf> {
        let df = df.sort(
            [self.args.duration_col],
            SortMultipleOptions::new().with_order_descending(false),
        )?;

        // This function should convert the DataFrame to the required format
        // and return a CoxProcessedDf with time_col, event_col, and covariates.
        // It should also sort the data and store the original indices.

        let time_array = df
            .column(self.args.duration_col)?
            .cast(&DataType::Float64)?
            .f64()?
            .into_no_null_iter()
            .collect::<Array1<f64>>();
        let event_array = df
            .column(self.args.event_col)?
            .cast(&DataType::Int32)? // Assuming event is 1 / 0
            .i32()?
            .into_no_null_iter()
            .collect::<Array1<i32>>();
        let covariates = df
            .drop_many(&[self.args.duration_col, self.args.event_col])
            .to_ndarray::<Float64Type>(IndexOrder::Fortran)?
            .into_shape((df.height(), df.width() - 2))
            .map_err(|e| anyhow!("Failed to convert DataFrame to 2D ndarray: {}", e))?;

        Ok(CoxProcessedDf {
            weights: Array1::<f64>::ones(time_array.len()),
            time_array,
            event_array,
            covariates,
        })
    }

    /// Returns
    /// - hessian (d, d) array
    /// - gradient (1, d) array
    /// - log_likelihood: f64
    ///
    /// Calculates the first and second order vector differentials, with respect to beta.
    /// Note that X, T, E are assumed to be sorted on T!
    ///
    ///  A good explanation for Efron. Consider three of five subjects who fail at the same time.
    ///  As it is not known a priori that who is the first to fail, so one-third of
    ///     (φ1 + φ2 + φ3) is adjusted from sum_j^{5} φj after one fails. Similarly two-third
    ///     of (φ1 + φ2 + φ3) is adjusted after first two individuals fail, etc.
    fn get_efron_values(
        &self,
        X: &Array2<f64>,
        T: &Array1<f64>,
        E: &Array1<i32>,
        weights: &Array1<f64>,
        beta: &Array1<f64>,
    ) -> (Array2<f64>, Array1<f64>, f64) {
        let shape = X.shape();
        let (n, d) = (shape[0], shape[1]);
        let mut hessian = Array2::<f64>::zeros((d, d));
        let mut gradient = Array1::<f64>::zeros(d);
        let mut log_lik = 0.0;

        // Init risk and tie sums to zero
        let mut x_death_sum = Array1::<f64>::zeros(d);
        let (mut risk_phi, mut tie_phi) = (0.0, 0.0);
        let (mut risk_phi_x, mut tie_phi_x) = (Array1::<f64>::zeros(d), Array1::<f64>::zeros(d));
        let (mut risk_phi_x_x, mut tie_phi_x_x) =
            (Array2::<f64>::zeros((d, d)), Array2::<f64>::zeros((d, d)));

        // Init number of ties and weights
        let mut weight_count = 0.0;
        let mut tied_death_counts = 0;
        let scores = math::exp(&(weights * X.dot(beta)));

        let phi_x_is = scores.clone().insert_axis(Axis(1)) * X;
        // let mut phi_x_x_i = Array2::<f64>::zeros((d, d));

        // Iterate backwards to utilize recursive relationship
        for i in n - 1..=0 {
            let ti = T[i];
            let ei = E[i];
            let xi = X.row(i);
            let w = weights[i];

            // Calculate phi values.
            let phi_i = scores[i];
            let phi_x_i = phi_x_is.row(i);
            let phi_x_x_i = math::outer_product(xi, phi_x_i);

            // Calculate sums of risk set
            risk_phi += &phi_i;
            risk_phi_x += &phi_x_i;
            risk_phi_x_x += &phi_x_x_i;

            // Calculate sums of Ties, if this is an event
            if ei != 0 {
                x_death_sum += &xi.mapv(|v| w * v);
                tie_phi += &phi_i;
                tie_phi_x += &phi_x_i;
                tie_phi_x_x += &phi_x_x_i;

                //  Keep track of count
                tied_death_counts += 1;
                weight_count += w;
            }

            if i > 0 && T[i - 1] == ti {
                // There are more ties/members of the risk set
                continue;
            } else if tied_death_counts == 0 {
                // Only censored with current time, move on
                continue;
            }

            // There was at least one event and no more ties remain. Time to sum.
            let weighted_average = weight_count / tied_death_counts as f64;

            if tied_death_counts > 1 {
                let increasing_proportion = Array1::<i32>::from_iter(0..tied_death_counts)
                    .mapv(|v| v as f64)
                    / tied_death_counts as f64;
                let denom = 1.0 / (risk_phi - &increasing_proportion * tie_phi);
                let numer = &risk_phi_x
                    - math::outer_product(increasing_proportion.view(), tie_phi_x.view());
                let a1 = (einsum("ab,i->ab", &[&risk_phi_x_x, &denom]).unwrap()
                    - einsum("ab,i->ab", &[&tie_phi_x_x, &increasing_proportion]).unwrap())
                .into_dimensionality::<Ix2>()
                .unwrap();

                let summand = numer * denom.clone().insert_axis(Axis(1));
                let a2 = summand.t().dot(&summand);

                gradient = gradient + &x_death_sum - weighted_average * summand.sum();
                log_lik = log_lik
                    + &x_death_sum.dot(beta)
                    + &weighted_average * &denom.mapv(|v| v.ln()).sum();
                hessian = hessian + weighted_average * (a2 - a1);
            } else {
                let denom = 1.0 / arr1(&[risk_phi]);
                let numer = &risk_phi_x;
                let a1 = &risk_phi_x_x * &denom;

                let summand = numer * &denom.clone().insert_axis(Axis(1));
                let a2 = summand.t().dot(&summand);

                gradient = gradient + &x_death_sum - weighted_average * summand.sum();
                log_lik = log_lik
                    + x_death_sum.dot(beta)
                    + weighted_average * &denom.mapv(|v| v.ln()).sum();
                hessian = hessian + weighted_average * (a2 - a1);
            };

            // reset tie values
            tied_death_counts = 0;
            weight_count = 0.0;
            x_death_sum.fill(0.0);
            tie_phi = 0.0;
            tie_phi_x.fill(0.0);
            tie_phi_x_x.fill(0.0);
        }

        (hessian, gradient, log_lik)
    }

    /// Newton Raphson algorithm for fitting CPH model.
    ///
    /// The data is assumed to be sorted on T!
    ///
    /// Return beta: (1,d) numpy array.
    fn newton_raphson_for_efron_model(
        &self,
        X: &Array2<f64>,
        T: &Array1<f64>,
        E: &Array1<i32>,
        weights: &Array1<f64>,
        step_size: f64,
        precision: f64,
        r_precision: f64,
        max_iter: usize,
    ) -> (Array1<f64>, f64, Array2<f64>) {
        let shape = X.shape();
        let (n, d) = (shape[0], shape[1]);

        let mut beta = Array1::<f64>::zeros(d);

        let mut step_sizer = StepSizer::new(step_size);
        let mut step_size = step_sizer.next();

        let mut delta = Array1::<f64>::zeros(d);
        let mut converging = true;
        let mut ll_ = 0.0;
        let mut previous_ll_ = 0.0;
        let mut i = 0; // iteration.
        let mut hessian = Array2::<f64>::zeros((d, d));

        while converging {
            beta.scaled_add(step_size, &delta);

            i += 1;

            // Because we do not have strata, we can directly get gradient from X, T, E, weights, entries, and beta.
            let (h, g, ll__) = self.get_efron_values(X, T, E, weights, &beta);
            ll_ = ll__;

            if self.args.penalizer > 0.0 {
                ll_ -= elastic_net_penalty(
                    &beta,
                    1.3f64.powf(i as _),
                    n as _,
                    self.args.penalizer,
                    self.args.l1_ratio,
                );
                // g -= d_elastic_net_penalty(beta, 1.3**i)
                // h[np.diag_indices(d)] -= dd_elastic_net_penalty(beta, 1.3**i)
            }

            

            let inv_h_dot_g_T = (-&h).solve(&g).unwrap();
            delta = inv_h_dot_g_T.clone();

            // save these as pending result
            hessian = h.clone();

            let norm_delta = if delta.len() > 0 { delta.norm() } else { 0.0 };

            // reusing an above piece to make g * inv(h) * g.T faster.
            let newton_decrement = g.dot(&inv_h_dot_g_T) / 2.0;

            if norm_delta < precision {
                converging = false;
            } else if previous_ll_ != 0.0
                && (ll_ - previous_ll_).abs() / (-previous_ll_) < r_precision
            {
                converging = false;
            } else if step_size <= 0.00001 {
                converging = false;
            } else if i >= max_iter {
                converging = false;
            } else if newton_decrement < precision {
                converging = false;
            }

            previous_ll_ = ll_;
            step_sizer.update(norm_delta);
            step_size = step_sizer.next();
        }

        (beta, ll_, hessian)
    }

    #[inline]
    pub fn new(args: CoxPHFitterArgs<'a>) -> Self {
        Self {
            args,
            time_to_event: None,
            event_indicator: None,
            covariates_matrix: None,
            sorted_original_indices: None,
        }
    }

    /// Fit the model. We use Efron's approximation that produces results closer to the exact combinatoric solution
    /// than Breslow's.
    pub fn fit(&self, df: &DataFrame) -> Result<CoxPHResults> {
        let cox_df = self.preprocess(df)?;

        let (beta, ll, hessian) = self.newton_raphson_for_efron_model(
            &cox_df.covariates,
            &cox_df.time_array,
            &cox_df.event_array,
            &cox_df.weights,
            0.95,
            1e-07,
            1e-9,
            self.args.max_iter,
        );

        println!("results: {beta}, {ll}, {hessian}");

        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_coxphfitter() {
        let data = LazyFrame::scan_parquet("./data/cox_data.parquet", Default::default())
            .unwrap()
            .collect()
            .unwrap();

        let args = CoxPHFitterArgs::default()
            .set_duration_col("T")
            .set_event_col("event")
            .set_penalizer(0.1)
            .set_robust(true);

        let cph = CoxPHFitter::new(args);
        cph.fit(&data).unwrap();
    }
}
